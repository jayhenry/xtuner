"""MoonEP's model-scoped XTuner integration.

The backend import remains lazy so unrelated dispatchers do not require
MoonEP. ``MoonEPModelRuntime`` owns model resources and ``MoonEPDispatcher``
owns one routed layer's static policy. One dispatch/combine call is a pure
data ``_MoonEPLayerCallState`` record advanced by the module-level transaction
functions (``dispatch_forward``, ``prepare_experts``, ``combine_forward``,
``combine_backward``, gradient completion, and the layer-Join handoff). The
private VMM workspace remains the deep module for physical expert layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import TypedDict, override

from xtuner.v1.ops.moe.cuda.route_weight import route_weight_rows_backward
from xtuner.v1.utils import log_rank0

from .base import ExpertWeightLayout, GenericDispatcher, PostDispatchResult, ProjectionPair
from .fsdp_vmm_landing import (
    accumulate_fsdp_unsharded_expert_gradients,
    fsdp_current_unsharded_expert_parameters,
    install_fsdp_vmm_landing,
    uninstall_fsdp_vmm_landing,
)
from .moonep_workspace import _ExpertVMMWorkspace


_INTEGRATION_API_VERSION = 3
_MOONEP_IMPORT_ERROR: ImportError | None

try:
    import moonep as _moonep_backend
except ImportError as exc:
    _moonep_backend = None
    _MOONEP_IMPORT_ERROR = exc
else:
    _MOONEP_IMPORT_ERROR = None


def require_moonep_backend() -> Any:
    """Validate the optional MoonEP-mod package when MoonEP is selected."""
    if _moonep_backend is None:
        raise RuntimeError("dispatcher='moonep' requires the MoonEP-mod integration package") from _MOONEP_IMPORT_ERROR

    source = getattr(_moonep_backend, "__file__", "<unknown>")
    if getattr(_moonep_backend, "XTUNER_INTEGRATION_API_VERSION", None) != _INTEGRATION_API_VERSION:
        raise RuntimeError(
            f"incompatible MoonEP integration API; expected {_INTEGRATION_API_VERSION}; loaded module: {source}"
        )
    return _moonep_backend


class MoonEPModelRuntime:
    """Own the resources shared by all routed layers in one model/EP group."""

    def __init__(
        self,
        *,
        ep_group: dist.ProcessGroup,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        intra_layer_micro_batch: int,
        staging_reference: bool,
        num_sms: int = 64,
    ) -> None:
        require_moonep_backend()
        if intra_layer_micro_batch < 1:
            raise ValueError("intra_layer_micro_batch must be positive")

        # MoonEP keeps token counts device-resident. Triton already satisfies
        # that contract; grouped_gemm does so only with its CUTLASS backend.
        # Validate this once here instead of branching in every GMM call.
        from xtuner.v1.module.grouped_linear import moe_group_linear
        from xtuner.v1.ops.moe.cuda import cutlass_group_gemm

        if cutlass_group_gemm is not None and moe_group_linear.group_gemm is cutlass_group_gemm:
            from grouped_gemm import backend as grouped_gemm_backend

            if os.environ.get("GROUPED_GEMM_USE_CUTLASS") != "1" or not grouped_gemm_backend.use_cutlass:
                raise RuntimeError(
                    "MoonEP with grouped_gemm requires GROUPED_GEMM_USE_CUTLASS=1 before importing grouped_gemm"
                )

        self._ep_group = ep_group
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_experts = num_experts
        self._top_k = top_k
        self._intra_layer_micro_batch = intra_layer_micro_batch
        self._staging_reference = staging_reference
        self._num_sms = num_sms

        # Model-owned execution resources. Buffer is created lazily once S is
        # known; workspace/stream and FSDP bindings are installed as one unit.
        self._buffer: Any | None = None
        self._workspace: _ExpertVMMWorkspace | None = None
        self._comm_stream: torch.cuda.Stream | None = None
        self._fsdp_params: tuple[Any, ...] = ()

        # Physical routed layers in FSDP execution order. Each entry is
        # (layer FQN, (fused_w1w3, fused_w2), landing generation).
        self._layers: list[tuple[str, tuple[nn.Module, nn.Module], int]] = []
        # MoonEP allocates fixed-size communication buffers on the first call;
        # later calls must retain the same source-token count S per EP rank.
        self._fixed_tokens_per_rank: int | None = None
        self._closed = False

    def build_dispatcher(
        self,
        *,
        layer_fqn: str,
        projections: tuple[nn.Module, nn.Module],
    ) -> MoonEPDispatcher:
        """Register one physical routed layer in FSDP execution order."""
        if any(registered_fqn == layer_fqn for registered_fqn, _, _ in self._layers):
            raise ValueError(f"duplicate MoonEP routed layer: {layer_fqn}")
        generation = len(self._layers) % 2
        self._layers.append((layer_fqn, projections, generation))
        return MoonEPDispatcher(
            runtime=self,
            layer_fqn=layer_fqn,
            projections=projections,
            generation=generation,
        )

    def validate_before_fsdp(self, fsdp_config: Any) -> None:
        """Validate the build-time FSDP policy without retaining its config."""
        if fsdp_config.param_dtype is not torch.bfloat16 or fsdp_config.reduce_dtype is not torch.bfloat16:
            raise ValueError("MoonEP requires BF16 FSDP param and reduce dtypes")
        if fsdp_config.cpu_offload:
            raise ValueError("MoonEP VMM weights cannot use FSDP CPU offload")
        if not fsdp_config.requires_grad:
            raise ValueError("MoonEP v1 requires trainable FSDP parameters")
        if not fsdp_config.reshard_after_forward:
            raise ValueError("MoonEP requires reshard_after_forward=True")

    def install_after_fsdp(self, *, fsdp_root: nn.Module) -> None:
        """Allocate execution resources after native FSDP has been
        installed."""
        if self._workspace is not None:
            raise RuntimeError("MoonEP FSDP resources are already installed")
        if not self._layers:
            raise TypeError("MoonEP requires at least one physical routed-expert layer")
        if self._staging_reference:
            log_rank0.warning(
                "moonep_staging_reference=True copies complete BF16 home expert "
                "weights after every FSDP AllGather; it is a numerical reference, "
                "not the production performance path."
            )
        workspace = _ExpertVMMWorkspace.allocate(
            projection_shapes=(
                (2 * self._intermediate_size, self._hidden_size),
                (self._hidden_size, self._intermediate_size),
            ),
            num_experts=self._num_experts,
            ep_group=self._ep_group,
            gradient_slots=self._intra_layer_micro_batch,
        )
        # Keep MoonEP collectives in FSDP's device-side launch order. A
        # separate high-priority stream forms an orthogonal progress wave with
        # NCCL and stalls at MoonEP's rank barriers under a full model.
        comm_stream = torch.cuda.current_stream()
        if not self._staging_reference:
            try:
                self._fsdp_params = install_fsdp_vmm_landing(
                    fsdp_root=fsdp_root,
                    targets=tuple(
                        (
                            layer_fqn,
                            projections,
                            workspace.landing(generation),
                        )
                        for layer_fqn, projections, generation in self._layers
                    ),
                )
            except Exception:
                workspace.destroy()
                raise
        self._workspace = workspace
        self._comm_stream = comm_stream

    def _validate_tokens_per_rank(self, tokens_per_rank: int) -> None:
        if self._closed:
            raise RuntimeError("MoonEP runtime was closed")
        if self._fixed_tokens_per_rank is None:
            self._fixed_tokens_per_rank = tokens_per_rank
        elif tokens_per_rank != self._fixed_tokens_per_rank:
            raise RuntimeError(f"MoonEP fixed S changed: {self._fixed_tokens_per_rank} -> {tokens_per_rank}")

    def _buffer_for(self, tokens_per_rank: int) -> Any:
        self._validate_tokens_per_rank(tokens_per_rank)
        if self._workspace is None:
            raise RuntimeError("MoonEP FSDP resources must be installed before forward")
        if self._buffer is None:
            assert _moonep_backend is not None
            self._buffer = _moonep_backend.Buffer(
                S=tokens_per_rank,
                H=self._hidden_size,
                K=self._top_k,
                E=self._num_experts,
                num_ep_ranks=self._ep_group.size(),
                group=self._ep_group,
                explicitly_destroy=True,
                num_sms=self._num_sms,
            )
        return self._buffer

    def _enqueue(self, operation, *, inputs: tuple[torch.Tensor | None, ...] = ()):
        """Run one MoonEP transaction on XTuner's stream and return its
        event."""
        stream = self._comm_stream
        if stream is None:
            raise RuntimeError("MoonEP FSDP resources must be installed before execution")
        caller_stream = torch.cuda.current_stream()
        stream.wait_event(caller_stream.record_event())
        for tensor in inputs:
            if tensor is not None:
                tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            result = operation()
            done = stream.record_event()
        return result, done

    def close(self) -> None:
        """Release Buffer before VMM workspace at a coordinated boundary."""
        if self._closed:
            return
        if self._comm_stream is not None:
            self._comm_stream.synchronize()
            self._comm_stream = None
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        if self._fsdp_params:
            uninstall_fsdp_vmm_landing(self._fsdp_params)
            self._fsdp_params = ()
        if self._workspace is not None:
            self._workspace.destroy()
            self._workspace = None
        self._layers.clear()
        self._closed = True


# Dispatcher shape legend:
#   S: source tokens on this EP rank, K: routed experts per token,
#   NvS: MoonEP's padded VM-group rows, E: global experts, B=E/R: home
#   experts per EP rank, H: hidden size.


class MoonEPPreDispatchResult(TypedDict):
    """Stage 1: device-resident router-space inputs normalized for MoonEP."""

    hidden_states: torch.Tensor  # [S, H], BF16 source-token order.
    topk_ids: torch.Tensor  # [S, K], contiguous int32 global expert IDs.
    tokens_per_expert: torch.Tensor  # [E], contiguous int32 source histogram.
    # The call state is opaque control state for the remaining five stages;
    # it never crosses into the compiled tensor-only expert block.
    _moonep_call: _MoonEPLayerCallState


class MoonEPDispatchResult(TypedDict):
    """Stage 2: global dispatch outputs plus its eager control-plane state."""

    hidden_states: torch.Tensor  # [NvS, H], BF16 physical VM-group order.
    topk_weights: torch.Tensor  # [NvS], FP32 weights in the same row order.
    # [E+B], int32 padded group ends; stays on device and is non-differentiable.
    cu_seqlens: torch.Tensor
    # Opaque per-call state shared by later stages; never enters compiled expert compute.
    _moonep_call: _MoonEPLayerCallState


class MoonEPPostDispatchResult(PostDispatchResult):
    """Stage 3: tensor-only local ``[2B]`` expert-compute bundle.

    ``hidden_states`` is ``[NvS, H]``; ``tokens_per_expert`` is device int32
    ``[2B]`` for home then duplicate groups; ``expert_weight_layout`` holds
    projection-paired ``[2B, O_p, I_p]`` weights and direct BF16 WGrad targets.
    """


class MoonEPPreCombineResult(TypedDict):
    """Stage 4: expert outputs before route scaling and reverse transport."""

    hidden_states: torch.Tensor  # [NvS, H], physical VM-group order.


class MoonEPCombineResult(TypedDict):
    """Stage 5: fused route-scaled output restored to source-token order."""

    hidden_states: torch.Tensor  # [S, H].


class MoonEPPostCombineResult(TypedDict):
    """Stage 6: final tensor bundle returned through the generic interface."""

    hidden_states: torch.Tensor  # [S, H].


@dataclass
class _MoonEPLayerGradients:
    """One FSDP call's initialization state, never shared across replay calls.

    Storage belongs to the runtime. Only the first backward producer clears it; forward/checkpoint replay must not
    touch a different call's live H.
    """

    initialized: bool = False


@dataclass(eq=False)
class _MoonEPLayerCallState:
    """One routed-layer call's pure lifecycle state for the transaction
    functions.

    The record carries borrowed model resources, the layer projections, and
    the call-local plan/event/weight/gradient handles that the module-level
    transaction functions read and advance. It owns no behavior and never
    references ``MoonEPDispatcher``. Identity, not field equality, distinguishes
    two calls, so instances stay hashable by ``id``.
    """

    runtime: MoonEPModelRuntime
    layer_fqn: str
    projections: tuple[nn.Module, nn.Module]
    generation: int
    grad_slot: int
    layer_gradients: _MoonEPLayerGradients

    # One MoonEP communication plan and its device-side dependency chain. Each
    # event is recorded once its named producer has been enqueued.
    plan: Any | None = None
    dispatch_done: Any | None = None
    weights_ready: Any | None = None
    combine_done: Any | None = None

    # Borrowed VMM aliases for this call, each [2B, O_p, I_p]. Gradient views
    # share home storage and must never be GEMM overwrite targets.
    local_weights: ProjectionPair | None = None
    fallback_gradient_targets: ProjectionPair | None = None

    # Current FSDP unsharded home Parameters [B, O_p, I_p] receive the returned
    # BF16 home gradients after both local projections complete.
    home_parameters: tuple[nn.Parameter, nn.Parameter] | None = None
    # Completed home views and the event covering the pair reduction.
    gradient_completion: tuple[ProjectionPair, Any] | None = None


# --- Transaction functions --------------------------------------------------
#
# Each function covers one complete device-side sequence for a single call and
# advances ``_MoonEPLayerCallState`` in place. The autograd Functions and the
# Dispatcher are the only callers; the pair backward must reuse the forward
# plan and gradient slot recorded on the call state.


def begin_dispatch(
    state: _MoonEPLayerCallState,
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk_weights: torch.Tensor,
    async_op: bool,
) -> MoonEPDispatchResult:
    """Create the activation autograd edge and start weight prefetch."""
    hidden_nvsh, weights_nvs, cu_seqlens = _DispatchAutograd.apply(
        hidden_states,
        topk_ids,
        tokens_per_expert,
        topk_weights,
        state,
        async_op,
    )
    return MoonEPDispatchResult(
        hidden_states=hidden_nvsh,
        topk_weights=weights_nvs,
        cu_seqlens=cu_seqlens,
        _moonep_call=state,
    )


def begin_combine(
    state: _MoonEPLayerCallState,
    *,
    expert_output: torch.Tensor,
    route_weights: torch.Tensor,
    async_op: bool,
) -> torch.Tensor:
    """Create the fused route-scaled combine autograd edge."""
    return _CombineAutograd.apply(expert_output, route_weights, state, async_op)


def finish_combine(state: _MoonEPLayerCallState, combined: torch.Tensor, *, async_op: bool) -> torch.Tensor:
    """Establish the final device dependency for an async combine."""
    if async_op:
        assert state.combine_done is not None
        state.combine_done.wait()
    return combined


def prepare_experts(state: _MoonEPLayerCallState, dispatched: MoonEPDispatchResult) -> MoonEPPostDispatchResult:
    """Wait at the first weight consumer and expose the tensor-only layout."""
    workspace = state.runtime._workspace
    assert workspace is not None
    assert state.weights_ready is not None
    assert state.local_weights is not None

    with torch.profiler.record_function("MoonEP::prepare_experts"):
        # This inserts a device dependency; it never waits on the host.
        state.weights_ready.wait()
        local_counts = workspace.local_token_counts(dispatched["cu_seqlens"])
        covered_rows = local_counts.sum()
        row_is_covered = (
            torch.arange(
                dispatched["hidden_states"].shape[0],
                device=dispatched["hidden_states"].device,
            )
            < covered_rows
        )
        hidden_states = dispatched["hidden_states"] * row_is_covered.unsqueeze(-1)
        local_counts = torch.cat(
            (
                local_counts[:-1],
                local_counts[-1:] + dispatched["hidden_states"].shape[0] - covered_rows,
            )
        )

    local_weights = state.local_weights
    state.local_weights = None
    # Keep the allocation-return GEMM interface. Joining activation and both
    # weight edges makes staging precede upstream activation backward; a
    # weight-only hook cannot establish that dependency.
    hidden_states, w13, w2 = _MoonEPExpertGradBridge.apply(
        hidden_states, nn.Parameter(local_weights[0]), nn.Parameter(local_weights[1]), state
    )
    return MoonEPPostDispatchResult(
        hidden_states=hidden_states,
        tokens_per_expert=local_counts,
        expert_weight_layout=ExpertWeightLayout(
            trainable_weights=(w13, w2),
        ),
    )


def current_home_parameters(state: _MoonEPLayerCallState) -> tuple[nn.Parameter, nn.Parameter]:
    """Return current FSDP leaves, staging into the landing only in reference
    mode."""
    runtime = state.runtime
    if not runtime._staging_reference:
        return fsdp_current_unsharded_expert_parameters(state.projections)

    workspace = runtime._workspace
    assert workspace is not None
    parameters: list[nn.Parameter] = []
    for linear, landing in zip(
        state.projections,
        workspace.landing(state.generation),
        strict=True,
    ):
        weight = cast(torch.Tensor, linear.weight)
        if not isinstance(weight, nn.Parameter):
            raise RuntimeError(f"{state.layer_fqn} staging expected an unsharded expert Parameter")
        source = weight.to_local() if isinstance(weight, DTensor) else weight
        if source.dtype is not torch.bfloat16 or source.numel() != landing.numel():
            raise RuntimeError(f"{state.layer_fqn} staging expected an unsharded BF16 expert weight")
        with torch.no_grad():
            landing.copy_(source.view_as(landing))
        parameters.append(weight)
    return parameters[0], parameters[1]


def dispatch_forward(
    state: _MoonEPLayerCallState,
    source_hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    source_route_weights: torch.Tensor,
    *,
    async_op: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch on a fresh plan and start both projection weight prefetches."""
    runtime = state.runtime
    buffer = runtime._buffer_for(source_hidden.shape[0])
    workspace = runtime._workspace
    assert workspace is not None

    def dispatch_and_prefetch():
        # The staging copy precedes dispatch's device barrier. The fresh plan
        # then starts both projection prefetches in this transaction.
        state.home_parameters = current_home_parameters(state)
        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
            source_hidden,
            route_weights_sk=source_route_weights,
            topk_experts_sk=topk_ids,
            tokens_per_expert=tokens_per_expert,
            async_finish=False,
            zero_copy=False,
        )
        assert route_weights_nvs is not None and cu_seqlens is not None
        state.plan = plan
        state.dispatch_done = torch.cuda.current_stream().record_event()
        state.local_weights, _ = workspace.prefetch_weights(
            buffer=buffer,
            plan=plan,
            generation=state.generation,
            grad_slot=state.grad_slot,
        )
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    with torch.profiler.record_function("MoonEP::dispatch_forward"):
        result, state.weights_ready = runtime._enqueue(
            dispatch_and_prefetch,
            inputs=(source_hidden, source_route_weights, topk_ids, tokens_per_expert),
        )
    assert state.dispatch_done is not None
    if not async_op:
        state.dispatch_done.wait()
    return result


def dispatch_backward(
    state: _MoonEPLayerCallState,
    grad_hidden_nvsh: torch.Tensor,
    grad_route_weights_nvs: torch.Tensor,
) -> ProjectionPair:
    """Combine activation and route-weight gradients on the saved plan."""
    runtime = state.runtime
    buffer = runtime._buffer
    assert state.plan is not None and buffer is not None
    grad_hidden_nvsh = grad_hidden_nvsh.contiguous()
    grad_route_weights_nvs = grad_route_weights_nvs.contiguous()

    def combine_gradients():
        grad_hidden, grad_route_weights, no_event = buffer.combine(
            plan=state.plan,
            hidden_nvsh=grad_hidden_nvsh,
            route_weights_nvs=grad_route_weights_nvs,
            async_finish=False,
            zero_copy=False,
        )
        assert grad_route_weights is not None and no_event is None
        return grad_hidden, grad_route_weights

    with torch.profiler.record_function("MoonEP::dispatch_backward"):
        result, done = runtime._enqueue(
            combine_gradients,
            inputs=(grad_hidden_nvsh, grad_route_weights_nvs),
        )
        done.wait()
    return result


def combine_forward(
    state: _MoonEPLayerCallState,
    expert_output: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    async_op: bool,
) -> torch.Tensor:
    """Fuse route scaling into the combine boundary on the saved plan."""
    runtime = state.runtime
    buffer = runtime._buffer
    assert state.plan is not None and buffer is not None

    def combine_output():
        output, gathered_weights, no_event = buffer.combine(
            plan=state.plan,
            hidden_nvsh=expert_output,
            hidden_scales_nvs=route_weights,
            route_weights_nvs=None,
            async_finish=False,
            zero_copy=False,
        )
        assert gathered_weights is None and no_event is None
        return output

    with torch.profiler.record_function("MoonEP::combine_forward"):
        output, state.combine_done = runtime._enqueue(
            combine_output,
            inputs=(expert_output, route_weights),
        )
    if not async_op:
        state.combine_done.wait()
    return output


def combine_backward(state: _MoonEPLayerCallState, grad_output: torch.Tensor) -> tuple[torch.Tensor, Any]:
    """Replay duplicated weights on the saved plan and return weighted grad."""
    runtime = state.runtime
    buffer = runtime._buffer
    workspace = runtime._workspace
    assert state.plan is not None and buffer is not None and workspace is not None
    grad_output = grad_output.contiguous()

    def dispatch_gradient_and_prefetch():
        # FSDP pre-backward has restored this generation. Stage it before
        # dispatch's barrier, then replay remote weights on the same stream.
        replay_home_parameters = current_home_parameters(state)
        if state.home_parameters is None:
            raise RuntimeError("MoonEP backward has no forward home Parameters")
        if any(
            replay is not forward
            for replay, forward in zip(replay_home_parameters, state.home_parameters, strict=True)
        ):
            raise RuntimeError("MoonEP backward observed a different FSDP unsharded Parameter")
        grad_weighted, no_weights, no_cu, reused_plan = buffer.dispatch(
            grad_output,
            plan=state.plan,
            async_finish=False,
            zero_copy=False,
        )
        assert no_weights is None and no_cu is None and reused_plan is state.plan
        gradient_dispatch_done = torch.cuda.current_stream().record_event()
        _, state.fallback_gradient_targets = workspace.prefetch_weights(
            buffer=buffer,
            plan=state.plan,
            generation=state.generation,
            grad_slot=state.grad_slot,
        )
        return grad_weighted, gradient_dispatch_done

    with torch.profiler.record_function("MoonEP::combine_backward"):
        (grad_weighted, gradient_dispatch_done), replay_done = runtime._enqueue(
            dispatch_gradient_and_prefetch,
            inputs=(grad_output,),
        )
        # Route-scale backward overlaps weight replay but cannot read the
        # dispatched gradient before this device event.
        gradient_dispatch_done.wait()
    return grad_weighted, replay_done


def start_gradient_completion(state: _MoonEPLayerCallState, gradients: ProjectionPair) -> None:
    """Stage allocation-return dW and enqueue the paired duplicate return."""
    if state.gradient_completion is not None:
        raise RuntimeError("MoonEP gradient completion was started twice")
    if state.plan is None or state.fallback_gradient_targets is None:
        raise RuntimeError("MoonEP gradient completion has no plan/targets")
    # Replay supplies fresh aliases without retaining forward versions.
    reduction_pair = state.fallback_gradient_targets
    runtime = state.runtime
    workspace = runtime._workspace
    buffer = runtime._buffer
    if workspace is None or buffer is None:
        raise RuntimeError("MoonEP gradient completion has no runtime workspace")

    def stage_and_reduce():
        # All read/modify/write operations on shared H run on the same runtime
        # stream, including the remote return kernel. A final event wait alone
        # cannot prevent concurrent writers racing here.
        b = runtime._num_experts // runtime._ep_group.size()
        for target, gradient in zip(reduction_pair, gradients, strict=True):
            if not state.layer_gradients.initialized:
                target[:b].zero_()
            target[:b].add_(gradient[:b])
            target[b:].copy_(gradient[b:])
        state.layer_gradients.initialized = True
        return workspace.complete_gradients(
            buffer=buffer,
            plan=state.plan,
            local_grads=reduction_pair,
            grad_slot=state.grad_slot,
        )

    with torch.profiler.record_function("MoonEP::gradient_handoff"):
        home_grads, done = runtime._enqueue(
            stage_and_reduce,
            inputs=(*reduction_pair, *gradients),
        )
    state.gradient_completion = (home_grads, done)


def finish_gradient_completion(
    state: _MoonEPLayerCallState,
) -> tuple[tuple[nn.Parameter, nn.Parameter], ProjectionPair]:
    """Wait on the device event; the layer Join owns the single handoff."""
    completion = state.gradient_completion
    if completion is None:
        raise RuntimeError("MoonEP gradient completion was not started")
    home_grads, done = completion
    done.wait()

    if state.home_parameters is None:
        raise RuntimeError("MoonEP gradient completion has no home Parameters")
    home_parameters = state.home_parameters

    state.fallback_gradient_targets = None
    state.home_parameters = None
    state.gradient_completion = None
    return home_parameters, home_grads


class _MoonEPExpertGradBridge(torch.autograd.Function):
    """Consume both dWs before releasing the expert activation gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        call_state: _MoonEPLayerCallState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.call_state = call_state
        return hidden_states, w13, w2

    @staticmethod
    def backward(ctx: Any, grad_hidden: torch.Tensor, dw13: torch.Tensor, dw2: torch.Tensor):
        start_gradient_completion(cast(_MoonEPLayerCallState, ctx.call_state), (dw13, dw2))
        # dW is now owned by MoonEP; do not also accumulate it on anchor leaves.
        return grad_hidden, None, None, None


class _MoonEPLayerGradJoin(torch.autograd.Function):
    """Join every microbatch before the native FSDP input backward hook."""

    @staticmethod
    def forward(
        ctx: Any,
        call_states: tuple[_MoonEPLayerCallState, ...],
        *layer_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.call_states = call_states
        return layer_inputs

    @staticmethod
    def backward(ctx: Any, *grad_inputs: torch.Tensor):
        # Event.wait inserts a dependency into the current CUDA stream; it does
        # not block the Python host or poll event readiness.
        for call_state in ctx.call_states:
            home_parameters, home_grads = finish_gradient_completion(call_state)
        # All return views alias the same H. Publishing each would count the
        # entire sum repeatedly. Native FSDP consumes this once via copy-in
        # before upstream calls may reuse H; RS itself may remain asynchronous.
        accumulate_fsdp_unsharded_expert_gradients(home_parameters, home_grads)
        return (None, *grad_inputs)


class _DispatchAutograd(torch.autograd.Function):
    """Bridge the dispatch/combine pair into PyTorch autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        source_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        source_route_weights: torch.Tensor,
        call_state: _MoonEPLayerCallState,
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.call_state = call_state
        hidden_nvsh, route_weights_nvs, cu_seqlens = dispatch_forward(
            call_state,
            source_hidden,
            topk_ids,
            tokens_per_expert,
            source_route_weights,
            async_op=async_op,
        )
        ctx.mark_non_differentiable(cu_seqlens)
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    @staticmethod
    def backward(
        ctx: Any,
        grad_hidden_nvsh: torch.Tensor,
        grad_route_weights_nvs: torch.Tensor,
        grad_cu_seqlens: None,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, None, None]:
        del grad_cu_seqlens
        grad_hidden, grad_route_weights = dispatch_backward(
            cast(_MoonEPLayerCallState, ctx.call_state),
            grad_hidden_nvsh,
            grad_route_weights_nvs,
        )
        return grad_hidden, None, None, grad_route_weights, None, None


class _CombineAutograd(torch.autograd.Function):
    """Bridge fused combine and saved-plan dispatch into autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        expert_output: torch.Tensor,
        route_weights: torch.Tensor,
        call_state: _MoonEPLayerCallState,
        async_op: bool,
    ) -> torch.Tensor:
        ctx.call_state = call_state
        ctx.save_for_backward(expert_output, route_weights)
        return combine_forward(call_state, expert_output, route_weights, async_op=async_op)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        grad_weighted, replay_done = combine_backward(cast(_MoonEPLayerCallState, ctx.call_state), grad_output)
        expert_output, route_weights = ctx.saved_tensors
        grad_expert, grad_route_weights = route_weight_rows_backward(
            grad_weighted,
            expert_output,
            route_weights,
        )
        # The next autograd node immediately reads duplicated weights.
        replay_done.wait()
        return grad_expert, grad_route_weights, None, None


class MoonEPDispatcher(
    GenericDispatcher[
        MoonEPPreDispatchResult,
        MoonEPDispatchResult,
        MoonEPPostDispatchResult,
        MoonEPPreCombineResult,
        MoonEPCombineResult,
        MoonEPPostCombineResult,
    ]
):
    """Adapt one routed layer to XTuner's six-stage dispatcher interface.

    This class owns only layer-static policy. Every dispatch creates a fresh
    ``_MoonEPLayerCallState`` that the module-level transaction functions
    advance with the call's plan, event, weight, and gradient state.
    """

    def __init__(
        self,
        *,
        runtime: MoonEPModelRuntime,
        layer_fqn: str,
        projections: tuple[nn.Module, nn.Module],
        generation: int,
    ) -> None:
        super().__init__(
            n_routed_experts=runtime._num_experts,
            process_group=runtime._ep_group,
        )
        self._runtime = runtime
        self._layer_fqn = layer_fqn
        self._projections = projections
        self._generation = generation
        self._next_gradient_slot = 0

    def _new_call_state(self, layer_gradients: _MoonEPLayerGradients) -> _MoonEPLayerCallState:
        """Allocate the next call-local slot and call-state token."""
        grad_slot = self._next_gradient_slot
        self._next_gradient_slot = (grad_slot + 1) % self._runtime._intra_layer_micro_batch
        return _MoonEPLayerCallState(
            runtime=self._runtime,
            layer_fqn=self._layer_fqn,
            projections=self._projections,
            generation=self._generation,
            grad_slot=grad_slot,
            layer_gradients=layer_gradients,
        )

    @override
    def prepare_layer_inputs(
        self,
        layer_inputs: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[object | None]]:
        """Create one call-local Join and one plan/duplicate slot per
        branch."""
        gradients = _MoonEPLayerGradients()
        call_states = tuple(self._new_call_state(gradients) for _ in layer_inputs)
        # No-grad original forwards build no backward node and never clear H.
        return list(_MoonEPLayerGradJoin.apply(call_states, *layer_inputs)), list(call_states)

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        layer_state: object | None = None,
        async_op: bool = False,
    ) -> MoonEPPreDispatchResult:
        del topk_weights, async_op
        if layer_state is None:
            raise RuntimeError("MoonEP dispatch_preprocess requires layer_state from prepare_layer_inputs")
        if not isinstance(layer_state, _MoonEPLayerCallState):
            raise TypeError("MoonEP layer_state must be a _MoonEPLayerCallState")
        self._runtime._validate_tokens_per_rank(hidden_states.shape[0])
        return MoonEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(dtype=torch.int32).contiguous(),
            tokens_per_expert=tokens_per_expert.to(dtype=torch.int32).contiguous(),
            _moonep_call=layer_state,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPDispatchResult:
        if decoding:
            raise NotImplementedError("MoonEP fixed-S training dispatch does not implement decoding")
        return begin_dispatch(
            pre_dispatched["_moonep_call"],
            hidden_states=pre_dispatched["hidden_states"],
            topk_ids=pre_dispatched["topk_ids"],
            tokens_per_expert=pre_dispatched["tokens_per_expert"],
            topk_weights=topk_weights.to(dtype=torch.float32).contiguous(),
            async_op=async_op,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        async_op: bool = False,
    ) -> MoonEPPostDispatchResult:
        del pre_dispatched, async_op
        return prepare_experts(dispatched["_moonep_call"], dispatched)

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPPreCombineResult:
        del pre_dispatched, dispatched, post_dispatched, async_op, decoding
        return MoonEPPreCombineResult(hidden_states=hidden_states)

    @override
    def combine(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        pre_combined: MoonEPPreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPCombineResult:
        del pre_dispatched, post_dispatched, decoding
        return MoonEPCombineResult(
            hidden_states=begin_combine(
                dispatched["_moonep_call"],
                expert_output=pre_combined["hidden_states"],
                route_weights=dispatched["topk_weights"],
                async_op=async_op,
            )
        )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        pre_combined: MoonEPPreCombineResult,
        combined: MoonEPCombineResult,
        async_op: bool = False,
    ) -> MoonEPPostCombineResult:
        del pre_dispatched, post_dispatched, pre_combined
        return MoonEPPostCombineResult(
            hidden_states=finish_combine(
                dispatched["_moonep_call"],
                combined["hidden_states"],
                async_op=async_op,
            )
        )


__all__ = [
    "MoonEPDispatcher",
    "MoonEPModelRuntime",
    "MoonEPPreDispatchResult",
    "MoonEPDispatchResult",
    "MoonEPPostDispatchResult",
    "MoonEPPreCombineResult",
    "MoonEPCombineResult",
    "MoonEPPostCombineResult",
    "require_moonep_backend",
]
