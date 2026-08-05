"""MoonEP's model-scoped XTuner integration.

The backend import lives in this module and remains lazy: importing XTuner or
building another dispatcher must not require MoonEP.  The three stateful
classes added here are intentionally deep modules.  ``MoonEPRuntime`` owns
model resources, ``_MoonEPInvocation`` owns one dispatch/combine pairing, and
``MoonEPDispatcher`` adapts that state to XTuner's six-stage dispatcher seam.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal, TypeAlias

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import TypedDict, override

from xtuner.v1.utils import log_rank0

from .base import GenericDispatcher


_INTEGRATION_API_VERSION = 1
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
        del fully_sharded_model  # FSDP still owns all checkpoint/parameter identity.
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
        if not staging_reference:
            # Issue 05 replaces this boundary with the versioned direct FSDP
            # landing adapter. Never silently pay for a staging copy.
            raise RuntimeError(
                "MoonEP direct FSDP landing is not installed; set "
                "moonep_staging_reference=True for the explicit reference path"
            )

        log_rank0.warning(
            "moonep_staging_reference=True copies complete BF16 home expert "
            "weights after every FSDP AllGather; it is a numerical reference, "
            "not the production performance path."
        )
        self._workspace = self._backend.ExpertVMMWorkspace.allocate(
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
            )
        return self._buffer

    def destroy(self) -> None:
        """Release Buffer before VMM workspace at a coordinated boundary."""
        if self._destroyed:
            return
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        if self._workspace is not None:
            self._workspace.destroy()
            self._workspace = None
        self._destroyed = True


ProjectionPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
ExpertTensorBundle: TypeAlias = tuple[ProjectionPair, ProjectionPair]


class MoonEPPreDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    tokens_per_expert: torch.Tensor


class MoonEPDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    cu_seqlens: torch.Tensor
    _moonep_invocation: _MoonEPInvocation


class MoonEPPostDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    tokens_per_expert: torch.Tensor
    expert_tensors: ExpertTensorBundle


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

    def dispatch(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        topk_weights: torch.Tensor,
        async_op: bool,
    ) -> MoonEPDispatchResult:
        buffer = self._runtime._buffer_for(hidden_states.shape[0])
        hidden_nvsh, weights_nvs, cu_seqlens, plan, done = buffer.dispatch(
            hidden_states,
            route_weights_sk=topk_weights,
            topk_experts_sk=topk_ids,
            tokens_per_expert=tokens_per_expert,
            async_finish=True,
            zero_copy=False,
        )
        assert weights_nvs is not None and cu_seqlens is not None
        self._plan = plan
        self._dispatch_done = done
        if not async_op:
            done.wait()
        return MoonEPDispatchResult(
            hidden_states=hidden_nvsh,
            topk_weights=weights_nvs,
            cu_seqlens=cu_seqlens,
            _moonep_invocation=self,
        )

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

        # Staging happens only after the enclosing FSDP layer has exposed its
        # current BF16 unsharded Parameter views. The subsequent MoonEP
        # prefetch records a caller-stream event, so these D2D copies are
        # ordered before remote weights are read without a host wait.
        landings = workspace.landing(self._owner._generation)
        with torch.no_grad():
            for linear, landing in zip(
                (self._owner._experts.fused_w1w3, self._owner._experts.fused_w2),
                landings,
            ):
                source = linear.weight.to_local() if isinstance(linear.weight, DTensor) else linear.weight
                if source.dtype is not torch.bfloat16 or source.numel() != landing.numel():
                    raise RuntimeError(f"{self._owner.layer_fqn} staging expected an unsharded BF16 expert weight")
                landing.copy_(source.view_as(landing))

        local_weights, grad_outputs, weights_ready = workspace.materialize(
            buffer=buffer,
            plan=self._plan,
            generation=self._owner._generation,
            grad_slot=self._grad_slot,
        )
        # async_op=True leaves dispatch on MoonEP's comm stream. Its event is
        # a device dependency (never a host wait) that makes cu_seqlens safe
        # before the caller stream derives local grouped-GEMM counts, while
        # remote weight prefetch may continue on the comm stream.
        assert self._dispatch_done is not None
        self._dispatch_done.wait()
        local_counts = workspace.local_token_counts(dispatched["cu_seqlens"])
        weights_ready.wait()
        return MoonEPPostDispatchResult(
            hidden_states=dispatched["hidden_states"],
            tokens_per_expert=local_counts,
            expert_tensors=(local_weights, grad_outputs),
        )

    def combine(self, expert_output: torch.Tensor, route_weights: torch.Tensor, *, async_op: bool) -> torch.Tensor:
        assert self._runtime._buffer is not None and self._plan is not None
        output, gathered_weights, done = self._runtime._buffer.combine(
            plan=self._plan,
            hidden_nvsh=expert_output,
            hidden_scales_nvs=route_weights,
            route_weights_nvs=None,
            async_finish=True,
            zero_copy=False,
        )
        assert gathered_weights is None
        self._combine_done = done
        if not async_op:
            done.wait()
        return output

    def wait_combined(self) -> None:
        assert self._combine_done is not None
        self._combine_done.wait()

    def finish_forward_only(self) -> None:
        # Event waits above are device dependencies. Dropping Python plan
        # ownership must never query/synchronize route-dependent work.
        self._plan = None
        self._dispatch_done = None
        self._combine_done = None


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
            training_dtype="bf16",
            generate_dtype="bf16",
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
        async_op: bool = False,
    ) -> MoonEPPreDispatchResult:
        del topk_weights, async_op
        self._runtime._validate_tokens_per_rank(hidden_states.shape[0])
        return MoonEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(dtype=torch.int32).contiguous(),
            # Keep the upstream dispatcher contract. The count stays entirely
            # on device and is consumed by MoonEP's route planner.
            tokens_per_expert=torch.bincount(
                topk_ids.flatten(), minlength=self._runtime._num_experts
            ).to(dtype=torch.int32),
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
