"""MoonEP's model-scoped XTuner integration.

The backend import lives in this module and remains lazy: importing XTuner or
building another dispatcher must not require MoonEP.  The three stateful
classes added here are intentionally deep modules.  ``MoonEPRuntime`` owns
model resources, ``_MoonEPInvocation`` owns one dispatch/combine pairing, and
``MoonEPDispatcher`` adapts that state to XTuner's six-stage dispatcher seam.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import TypedDict, override

from xtuner.v1.ops.moe.cuda.route_weight import route_weight_rows_backward
from xtuner.v1.utils import log_rank0

from .base import ExpertWeightLayout, GenericDispatcher, PostDispatchResult, ProjectionPair
from .fsdp_vmm_landing import (
    fsdp_current_unsharded_expert_weights,
    install_fsdp_vmm_landing,
    uninstall_fsdp_vmm_landing,
)


_INTEGRATION_API_VERSION = 2
_TARGET_TORCH_VERSION = "2.12.1+cu132"


def require_moonep_backend() -> Any:
    """Load and validate the optional MoonEP-mod package on first selection."""
    try:
        backend = importlib.import_module("moonep")
    except ImportError as exc:
        raise RuntimeError("dispatcher='moonep' requires the MoonEP-mod integration package") from exc

    source = getattr(backend, "__file__", "<unknown>")
    if getattr(backend, "XTUNER_INTEGRATION_API_VERSION", None) != _INTEGRATION_API_VERSION:
        raise RuntimeError(
            f"incompatible MoonEP integration API; expected {_INTEGRATION_API_VERSION}; loaded module: {source}"
        )

    workspace = getattr(backend, "ExpertVMMWorkspace", None)
    if (
        not hasattr(backend, "Buffer")
        or workspace is None
        or not hasattr(workspace, "validate")
        or not hasattr(workspace, "allocate")
    ):
        raise RuntimeError(f"MoonEP-mod XTuner capabilities are missing: {source}")
    if torch.__version__ != _TARGET_TORCH_VERSION:
        raise RuntimeError(
            f"MoonEP integration requires torch {_TARGET_TORCH_VERSION}, "
            f"got {torch.__version__}; loaded module: {source}"
        )
    return backend


class MoonEPRuntime:
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
        num_sms: int = 64,
    ) -> None:
        self._backend = require_moonep_backend()
        if intra_layer_micro_batch < 1:
            raise ValueError("intra_layer_micro_batch must be positive")

        self._ep_group = ep_group
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_experts = num_experts
        self._top_k = top_k
        self._intra_layer_micro_batch = intra_layer_micro_batch
        self._num_sms = num_sms

        # This is deliberately the complete meta-build action.  The backend
        # validates kernel metadata here but cannot create CUDA/VMM/socket
        # resources until native FSDP has finished mutating parameters.
        self._backend.ExpertVMMWorkspace.validate(
            projection_shapes=(
                (2 * intermediate_size, hidden_size),
                (hidden_size, intermediate_size),
            ),
            num_experts=num_experts,
            ep_size=ep_group.size(),
            top_k=top_k,
            dtype=torch.bfloat16,
            home_generations=2,
            gradient_slots=intra_layer_micro_batch,
        )

        self._buffer: Any | None = None
        self._workspace: Any | None = None
        self._fsdp_params: tuple[Any, ...] = ()
        self._staging_reference = False
        self._dispatchers: list[Any] = []
        self._fixed_tokens_per_rank: int | None = None
        self._destroyed = False

    def dispatcher_for(
        self,
        *,
        layer_fqn: str,
        experts: nn.Module,
    ) -> MoonEPDispatcher:
        """Register one physical routed layer in FSDP execution order."""
        if any(dispatcher.layer_fqn == layer_fqn for dispatcher in self._dispatchers):
            raise ValueError(f"duplicate MoonEP routed layer: {layer_fqn}")
        generation: Literal[0, 1] = 0 if len(self._dispatchers) % 2 == 0 else 1
        dispatcher = MoonEPDispatcher(
            runtime=self,
            layer_fqn=layer_fqn,
            experts=experts,
            generation=generation,
        )
        self._dispatchers.append(dispatcher)
        return dispatcher

    def install_fsdp(
        self,
        *,
        fully_sharded_model: nn.Module,
        fsdp_config: Any,
        staging_reference: bool,
    ) -> None:
        """Allocate execution resources after native FSDP has been
        installed."""
        if self._workspace is not None:
            raise RuntimeError("MoonEP FSDP resources are already installed")
        if fsdp_config.param_dtype is not torch.bfloat16 or fsdp_config.reduce_dtype is not torch.bfloat16:
            raise ValueError("MoonEP requires BF16 FSDP param and reduce dtypes")
        if fsdp_config.cpu_offload:
            raise ValueError("MoonEP VMM weights cannot use FSDP CPU offload")
        if not fsdp_config.requires_grad:
            raise ValueError("MoonEP v1 requires trainable FSDP parameters")
        if not fsdp_config.reshard_after_forward:
            raise ValueError("MoonEP requires reshard_after_forward=True")
        if not self._dispatchers:
            raise TypeError("MoonEP requires at least one physical routed-expert layer")
        if staging_reference:
            log_rank0.warning(
                "moonep_staging_reference=True copies complete BF16 home expert "
                "weights after every FSDP AllGather; it is a numerical reference, "
                "not the production performance path."
            )
        workspace = self._backend.ExpertVMMWorkspace.allocate(
            projection_shapes=(
                (2 * self._intermediate_size, self._hidden_size),
                (self._hidden_size, self._intermediate_size),
            ),
            num_experts=self._num_experts,
            ep_group=self._ep_group,
            top_k=self._top_k,
            dtype=torch.bfloat16,
            home_generations=2,
            gradient_slots=self._intra_layer_micro_batch,
        )
        if not staging_reference:
            try:
                self._fsdp_params = install_fsdp_vmm_landing(
                    fully_sharded_model=fully_sharded_model,
                    targets=tuple(
                        (
                            dispatcher.layer_fqn,
                            dispatcher._experts,
                            workspace.landing(dispatcher._generation),
                        )
                        for dispatcher in self._dispatchers
                    ),
                )
            except Exception:
                workspace.destroy()
                raise
        self._staging_reference = staging_reference
        self._workspace = workspace

    def _validate_tokens_per_rank(self, tokens_per_rank: int) -> None:
        if self._destroyed:
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
            self._buffer = self._backend.Buffer(
                S=tokens_per_rank,
                H=self._hidden_size,
                K=self._top_k,
                E=self._num_experts,
                num_ep_ranks=self._ep_group.size(),
                group=self._ep_group,
                explicitly_destroy=True,
                num_sms=self._num_sms,
                # FSDP collectives use the caller stream.  Giving MoonEP the
                # same device-side launch order prevents orthogonal EP/FSDP
                # progress waves without introducing a host synchronization.
                use_caller_stream=True,
            )
        return self._buffer

    def destroy(self) -> None:
        """Release Buffer before VMM workspace at a coordinated boundary."""
        if self._destroyed:
            return
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        if self._fsdp_params:
            uninstall_fsdp_vmm_landing(self._fsdp_params)
            self._fsdp_params = ()
        if self._workspace is not None:
            self._workspace.destroy()
            self._workspace = None
        self._destroyed = True


class MoonEPPreDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    tokens_per_expert: torch.Tensor


class MoonEPDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    cu_seqlens: torch.Tensor
    _moonep_invocation: _MoonEPInvocation


class MoonEPPostDispatchResult(PostDispatchResult): ...


class MoonEPPreCombineResult(TypedDict):
    hidden_states: torch.Tensor
    _moonep_invocation: _MoonEPInvocation


class MoonEPCombineResult(TypedDict):
    hidden_states: torch.Tensor


class MoonEPPostCombineResult(TypedDict):
    hidden_states: torch.Tensor


class _MoonEPInvocation:
    """Own one fresh MoonEP plan and all of its device-side completion
    edges."""

    def __init__(self, owner: MoonEPDispatcher, *, grad_slot: int) -> None:
        self._owner = owner
        self._runtime = owner._runtime
        self._grad_slot = grad_slot
        self._plan: Any | None = None
        self._dispatch_done: Any | None = None
        self._combine_done: Any | None = None
        self._gradient_targets: ProjectionPair | None = None

    def dispatch(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        topk_weights: torch.Tensor,
        async_op: bool,
    ) -> MoonEPDispatchResult:
        hidden_nvsh, weights_nvs, cu_seqlens = _DispatchAutograd.apply(
            hidden_states,
            topk_ids,
            tokens_per_expert,
            topk_weights,
            self,
            async_op,
        )
        return MoonEPDispatchResult(
            hidden_states=hidden_nvsh,
            topk_weights=weights_nvs,
            cu_seqlens=cu_seqlens,
            _moonep_invocation=self,
        )

    def _current_home_weights(self) -> ProjectionPair:
        """Return this layer's current FSDP views, staging only by request."""
        if not self._runtime._staging_reference:
            return fsdp_current_unsharded_expert_weights(self._owner._experts)

        workspace = self._runtime._workspace
        assert workspace is not None
        landings = workspace.landing(self._owner._generation)
        sources: list[torch.Tensor] = []
        for linear, landing in zip(
            (self._owner._experts.fused_w1w3, self._owner._experts.fused_w2),
            landings,
        ):
            source = linear.weight.to_local() if isinstance(linear.weight, DTensor) else linear.weight
            if source.dtype is not torch.bfloat16 or source.numel() != landing.numel():
                raise RuntimeError(f"{self._owner.layer_fqn} staging expected an unsharded BF16 expert weight")
            with torch.no_grad():
                landing.copy_(source.view_as(landing))
            sources.append(source)
        return sources[0], sources[1]

    def prepare_experts(
        self,
        dispatched: MoonEPDispatchResult,
        *,
        async_op: bool,
    ) -> MoonEPPostDispatchResult:
        del async_op
        runtime = self._runtime
        workspace = runtime._workspace
        buffer = runtime._buffer
        assert workspace is not None and buffer is not None and self._plan is not None

        # FSDP has exposed differentiable BF16 unsharded views at this point.
        # Staging copies their values but keeps the source tensors as the
        # native FSDP gradient edge.
        with torch.profiler.record_function("MoonEP::prepare_experts"):
            home_weights = self._current_home_weights()
            local_weights, gradient_targets, weights_ready = workspace.materialize(
                buffer=buffer,
                plan=self._plan,
                generation=self._owner._generation,
                grad_slot=self._grad_slot,
            )
            self._gradient_targets = gradient_targets
            # async_op=True leaves dispatch on MoonEP's comm stream. Its event
            # is a device dependency (never a host wait) that makes cu_seqlens
            # safe before deriving local grouped-GEMM counts.
            assert self._dispatch_done is not None
            self._dispatch_done.wait()
            local_counts = workspace.local_token_counts(dispatched["cu_seqlens"])
            # MoonEP returns a fixed-capacity buffer while standard grouped
            # GEMM requires counts to cover every row. Assign the zeroed tail
            # to the final physical group so every backend sees one contract.
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
            weights_ready.wait()
        differentiable_weights = _ExpertWeightAutograd.apply(
            home_weights[0],
            home_weights[1],
            local_weights[0],
            local_weights[1],
            self,
        )
        return MoonEPPostDispatchResult(
            hidden_states=hidden_states,
            tokens_per_expert=local_counts,
            expert_weight_layout=ExpertWeightLayout(trainable_weights=differentiable_weights),
        )

    def combine(self, expert_output: torch.Tensor, route_weights: torch.Tensor, *, async_op: bool) -> torch.Tensor:
        return _CombineAutograd.apply(expert_output, route_weights, self, async_op)

    def wait_combined(self) -> None:
        assert self._combine_done is not None
        self._combine_done.wait()

    def finish_forward_only(self) -> None:
        # Event waits above are device dependencies. Dropping Python plan
        # ownership must never query/synchronize route-dependent work.
        self._plan = None
        self._dispatch_done = None
        self._combine_done = None
        self._gradient_targets = None

    def _dispatch_backward(
        self,
        grad_hidden_nvsh: torch.Tensor,
        grad_route_weights_nvs: torch.Tensor,
    ) -> ProjectionPair:
        """Use forward's plan to combine source hidden/router gradients."""
        assert self._plan is not None and self._runtime._buffer is not None
        with torch.profiler.record_function("MoonEP::dispatch_backward"):
            grad_hidden, grad_route_weights, done = self._runtime._buffer.combine(
                plan=self._plan,
                hidden_nvsh=grad_hidden_nvsh.contiguous(),
                route_weights_nvs=grad_route_weights_nvs.contiguous(),
                async_finish=True,
                zero_copy=False,
            )
            assert grad_route_weights is not None
            done.wait()
        return grad_hidden, grad_route_weights

    def _combine_backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Dispatch output gradients and overlap duplicated-weight replay."""
        runtime = self._runtime
        workspace = runtime._workspace
        assert self._plan is not None and runtime._buffer is not None and workspace is not None
        with torch.profiler.record_function("MoonEP::combine_backward"):
            grad_weighted, no_weights, no_cu, reused_plan, dispatch_done = runtime._buffer.dispatch(
                grad_output.contiguous(),
                plan=self._plan,
                async_finish=True,
                zero_copy=False,
            )
            assert no_weights is None and no_cu is None and reused_plan is self._plan

            # The pre-backward AllGather has restored this generation before
            # duplicated weights are replayed.
            self._current_home_weights()
            _, _, replay_done = workspace.materialize(
                buffer=runtime._buffer,
                plan=self._plan,
                generation=self._owner._generation,
                grad_slot=self._grad_slot,
            )
            dispatch_done.wait()
        return grad_weighted, replay_done

    def _complete_weight_gradients(self, local_grads: ProjectionPair) -> ProjectionPair:
        """Return duplicated BF16 partials before FSDP ReduceScatter."""
        runtime = self._runtime
        assert self._plan is not None and runtime._buffer is not None and runtime._workspace is not None
        assert self._gradient_targets is not None
        with torch.profiler.record_function("MoonEP::gradient_handoff"):
            # Standard one-segment GMMs own their natural dW allocation. Stage
            # that result into MoonEP's symmetric reduction slot privately;
            # the grouped-linear and op interfaces remain backend-neutral.
            for source, target in zip(local_grads, self._gradient_targets):
                target.copy_(source)
            home_grads, done = runtime._workspace.complete_gradients(
                buffer=runtime._buffer,
                plan=self._plan,
                local_grads=self._gradient_targets,
                grad_slot=self._grad_slot,
            )
            done.wait()
        self._gradient_targets = None
        return home_grads


class _DispatchAutograd(torch.autograd.Function):
    """Concrete MoonEP dispatch forward paired with combine backward."""

    @staticmethod
    def forward(
        ctx: Any,
        source_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        source_route_weights: torch.Tensor,
        invocation: _MoonEPInvocation,
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.invocation = invocation
        buffer = invocation._runtime._buffer_for(source_hidden.shape[0])
        with torch.profiler.record_function("MoonEP::dispatch_forward"):
            hidden_nvsh, route_weights_nvs, cu_seqlens, plan, done = buffer.dispatch(
                source_hidden,
                route_weights_sk=source_route_weights,
                topk_experts_sk=topk_ids,
                tokens_per_expert=tokens_per_expert,
                async_finish=True,
                zero_copy=False,
            )
        assert route_weights_nvs is not None and cu_seqlens is not None
        invocation._plan = plan
        invocation._dispatch_done = done
        if not async_op:
            done.wait()
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
        grad_hidden, grad_route_weights = ctx.invocation._dispatch_backward(
            grad_hidden_nvsh,
            grad_route_weights_nvs,
        )
        return grad_hidden, None, None, grad_route_weights, None, None


class _CombineAutograd(torch.autograd.Function):
    """Fused route-scaled combine forward paired with plan-reuse dispatch."""

    @staticmethod
    def forward(
        ctx: Any,
        expert_output: torch.Tensor,
        route_weights: torch.Tensor,
        invocation: _MoonEPInvocation,
        async_op: bool,
    ) -> torch.Tensor:
        ctx.invocation = invocation
        ctx.save_for_backward(expert_output, route_weights)
        assert invocation._plan is not None and invocation._runtime._buffer is not None
        with torch.profiler.record_function("MoonEP::combine_forward"):
            output, gathered_weights, done = invocation._runtime._buffer.combine(
                plan=invocation._plan,
                hidden_nvsh=expert_output,
                hidden_scales_nvs=route_weights,
                route_weights_nvs=None,
                async_finish=True,
                zero_copy=False,
            )
        assert gathered_weights is None
        invocation._combine_done = done
        if not async_op:
            done.wait()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        grad_weighted, replay_done = ctx.invocation._combine_backward(grad_output)
        expert_output, route_weights = ctx.saved_tensors
        grad_expert, grad_route_weights = route_weight_rows_backward(
            grad_weighted,
            expert_output,
            route_weights,
        )
        # The next autograd node immediately reads duplicated weights.
        replay_done.wait()
        return grad_expert, grad_route_weights, None, None


class _ExpertWeightAutograd(torch.autograd.Function):
    """Complete two local-[2B] weight gradients as one FSDP transaction."""

    @staticmethod
    def forward(
        ctx: Any,
        home_w1w3: torch.Tensor,
        home_w2: torch.Tensor,
        local_w1w3: torch.Tensor,
        local_w2: torch.Tensor,
        invocation: _MoonEPInvocation,
    ) -> ProjectionPair:
        ctx.invocation = invocation
        ctx.home_shapes = home_w1w3.shape, home_w2.shape
        return local_w1w3.view_as(local_w1w3), local_w2.view_as(local_w2)

    @staticmethod
    def backward(
        ctx: Any,
        grad_local_w1w3: torch.Tensor,
        grad_local_w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        home_w1w3_grad, home_w2_grad = ctx.invocation._complete_weight_gradients((grad_local_w1w3, grad_local_w2))
        return (
            home_w1w3_grad.reshape(ctx.home_shapes[0]),
            home_w2_grad.reshape(ctx.home_shapes[1]),
            None,
            None,
            None,
        )


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
    """Adapt MoonEP plans/VMM state to XTuner's six-stage dispatcher API."""

    def __init__(
        self,
        *,
        runtime: MoonEPRuntime,
        layer_fqn: str,
        experts: Any,
        generation: Literal[0, 1],
    ) -> None:
        super().__init__(
            n_routed_experts=runtime._num_experts,
            process_group=runtime._ep_group,
        )
        self._runtime = runtime
        self._layer_fqn = layer_fqn
        self._experts = experts
        self._generation = generation
        self._next_gradient_slot = 0

    @property
    def layer_fqn(self) -> str:
        return self._layer_fqn

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        async_op: bool = False,
    ) -> MoonEPPreDispatchResult:
        del topk_weights, async_op
        self._runtime._validate_tokens_per_rank(hidden_states.shape[0])
        return MoonEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(dtype=torch.int32).contiguous(),
            tokens_per_expert=tokens_per_expert.to(dtype=torch.int32).contiguous(),
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
        grad_slot = self._next_gradient_slot
        self._next_gradient_slot = (grad_slot + 1) % self._runtime._intra_layer_micro_batch
        return _MoonEPInvocation(self, grad_slot=grad_slot).dispatch(
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
        decoding: bool = False,
    ) -> MoonEPPostDispatchResult:
        del pre_dispatched, decoding
        return dispatched["_moonep_invocation"].prepare_experts(dispatched, async_op=async_op)

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
        del pre_dispatched, post_dispatched, async_op, decoding
        return MoonEPPreCombineResult(
            hidden_states=hidden_states,
            _moonep_invocation=dispatched["_moonep_invocation"],
        )

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
        invocation = pre_combined["_moonep_invocation"]
        return MoonEPCombineResult(
            hidden_states=invocation.combine(
                pre_combined["hidden_states"],
                dispatched["topk_weights"],
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
        del pre_dispatched, dispatched, post_dispatched
        invocation = pre_combined["_moonep_invocation"]
        if async_op:
            invocation.wait_combined()
        result = MoonEPPostCombineResult(hidden_states=combined["hidden_states"])
        if not torch.is_grad_enabled():
            invocation.finish_forward_only()
        return result


__all__ = [
    "MoonEPDispatcher",
    "MoonEPRuntime",
    "MoonEPPreDispatchResult",
    "MoonEPDispatchResult",
    "MoonEPPostDispatchResult",
    "MoonEPPreCombineResult",
    "MoonEPCombineResult",
    "MoonEPPostCombineResult",
    "require_moonep_backend",
]
