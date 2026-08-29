"""MoonEP / UltraEP 一致接入方案的结构化伪代码。

这不是可直接替换 XTuner 文件的 patch。它刻意展示：

1. concrete model-scoped runtimes；
2. per-layer Dispatcher Adapter 与已有六阶段；
3. predecessor-only Dispatcher state 与 backend-private state；
4. MLP-level expert weight layout 与 counts-based GroupedLinear；
5. 收窄后的 runtime / layer / autograd 参数边界；
6. BF16 / FP8 单段算子与 UltraEP 双段算子的兼容边界；
7. 单 microbatch、Domino、FSDP install 和 teardown 的主要调用端流程。

省略了真实 backend import、stream 细节、错误文本和现有 Dispatcher 的具体通信代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, NamedTuple, Sequence, TypeAlias, TypedDict, cast

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor


StageState = dict[str, Any]
DispatcherName = Literal["naive", "all2all", "deepep", "moonep", "ultraep"] | None


# =============================================================================
# 1. [CHANGED CONTRACT] Router 与 expert compute value contracts
# =============================================================================


class RouterResults(TypedDict):
    """Router 是 logical source counts 的唯一生产者。"""

    logits: Tensor
    router_weights: Tensor
    topk_ids: Tensor
    topk_weights: Tensor
    tokens_per_expert: Tensor  # [E] logical counts；替代 topkens_per_expert


ProjectionPair: TypeAlias = tuple[Tensor, Tensor]  # (fused_w1w3, fused_w2)


class ExpertWeightLayout(NamedTuple):
    """MLP-level tensor value; no runtime/plan/event crosses the compile Seam.

    Trainable weights always use natural autograd WGrad.  The optional external
    segment is runtime-owned and receives dW only through explicit output tensors.
    """

    trainable_weights: ProjectionPair | None = None
    external_weights: ProjectionPair | None = None
    external_wgrad_outs: ProjectionPair | None = None


# =============================================================================
# 2. [CHANGED BASE] 直接复用既有六阶段
# =============================================================================


class GenericDispatcher(ABC):
    """六阶段 predecessor-only state pipeline。

    ``prepare_layer_input`` 是 UltraEP backward ordering 所需的唯一
    pre-attention hook；默认返回 identity tensor 和空 state。它不包装、替代
    或缓存后续六阶段。Flags 只在 phase 1 写入 state。
    """

    def prepare_layer_input(self, layer_input: Tensor) -> tuple[Tensor, object | None]:
        return layer_input, None

    @abstractmethod
    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        layer_state: object | None,
        dispatch_async: bool = False,
        combine_async: bool = False,
        decoding: bool = False,
    ) -> StageState:
        """阶段 1；tokens_per_expert 是 Router-owned logical [E] counts。"""

    @abstractmethod
    def dispatch(
        self,
        state: StageState,
    ) -> StageState:
        """阶段 2；只消费 phase-1 state。"""

    @abstractmethod
    def dispatch_postprocess(
        self,
        state: StageState,
    ) -> StageState:
        """阶段 3；交付 local tensors 与 call-local ExpertWeightLayout。"""

    @abstractmethod
    def combine_preprocess(
        self,
        state: StageState,
        expert_output: Tensor,
    ) -> StageState:
        """阶段 4；state 是 phase-3 output。"""

    @abstractmethod
    def combine(
        self,
        state: StageState,
    ) -> StageState:
        """阶段 5；只消费 phase-4 state。"""

    @abstractmethod
    def combine_postprocess(
        self,
        state: StageState,
    ) -> Tensor:
        """阶段 6；返回 routed expert output。"""


# Existing Naive/All2All/DeepEP/AGRS Adapters only need three mechanical changes:
#
# 1. phase 1 stores flags/route metadata in its call-local state;
# 2. every later phase accepts only the predecessor state;
# 3. phase 3 returns ExpertWeightLayout() for the ordinary trainable-only path.
#
# Their communication Implementation and remaining five stages stay unchanged.


# =============================================================================
# 3. [MOONEP ADAPTER] 保留 Buffer / VMM / private autograd Implementation
# =============================================================================


class _MoonEPInvocation:
    """MoonEP per-call plan/events/reduction slot；真实逻辑复用当前实现。"""

    def dispatch(
        self,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        *,
        async_op: bool,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def materialize_expert_layout(
        self,
    ) -> tuple[Tensor, Tensor, ExpertWeightLayout]:
        """返回 dispatched hidden、local [2B] counts 和 MLP-level layout。"""
        hidden_states, local_counts, weights = materialize_moonep_local_tensors(self)
        return (
            hidden_states,
            local_counts,
            ExpertWeightLayout(
                # These are differentiable _ExpertWeightAutograd outputs.  The
                # selected standard GMM returns their dW through natural autograd.
                trainable_weights=weights,
            ),
        )

    def _complete_weight_gradients(self, local_grads: ProjectionPair) -> ProjectionPair:
        """Stage natural dW if needed, reduce duplicates, and return home dW。"""
        raise NotImplementedError

    def combine_preprocess(self, expert_output: Tensor, *, async_op: bool) -> Tensor:
        raise NotImplementedError

    def combine(
        self,
        hidden_states: Tensor,
        route_weights: Tensor,
        *,
        async_op: bool,
    ) -> Tensor:
        raise NotImplementedError

    def combine_postprocess(self, hidden_states: Tensor, *, async_op: bool) -> Tensor:
        raise NotImplementedError

    def finish_forward_only(self) -> None:
        raise NotImplementedError


class MoonEPRuntime:
    """现有 model-scoped MoonEP runtime；直接实现 model 使用的 lifecycle。"""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        max_inflight: int,
        num_sms: int,
        staging_reference: bool,
        ep_group: Any,
    ) -> None:
        # Full MoEConfig belongs to the model-build validation Seam. The runtime
        # retains only immutable facts needed by its resource/lifetime rules.
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_inflight = max_inflight
        self.num_sms = num_sms
        self.staging_reference = staging_reference
        self.ep_group = ep_group
        self._layers: list[tuple[str, tuple[nn.Module, nn.Module]]] = []
        self._closed = False

    def bind_dispatcher(
        self,
        *,
        layer_fqn: str,
        projections: tuple[nn.Module, nn.Module],
    ) -> GenericDispatcher:
        if any(registered_fqn == layer_fqn for registered_fqn, _ in self._layers):
            raise ValueError(f"duplicate MoonEP routed layer: {layer_fqn}")
        layer_id = len(self._layers)
        self._layers.append((layer_fqn, projections))
        return MoonEPDispatcher(runtime=self, layer_id=layer_id)

    def validate_before_fsdp(self, *, fsdp_config: Any) -> None:
        validate_moonep_fsdp_config(fsdp_config)

    def install_after_fsdp(self, *, fsdp_root: nn.Module) -> None:
        # The version-pinned FSDP2 Adapter needs a traversal root to locate owner
        # states. It is call-scoped and is never retained by the runtime.
        install_moonep_vmm_and_fsdp_landing(self, fsdp_root)

    def close(self) -> None:
        if not self._closed:
            destroy_moonep_buffers_vmm_and_landing(self)
            self._layers.clear()
            self._closed = True


class _ExpertWeightAutograd(torch.autograd.Function):
    """既有 MoonEP weight bridge；作用等价于成对的 local-weight grad hook。"""

    @staticmethod
    def forward(
        ctx: Any,
        home_w1w3: Tensor,
        home_w2: Tensor,
        local_w1w3: Tensor,
        local_w2: Tensor,
        invocation: _MoonEPInvocation,
    ) -> ProjectionPair:
        ctx.invocation = invocation
        ctx.home_shapes = home_w1w3.shape, home_w2.shape
        return local_w1w3.view_as(local_w1w3), local_w2.view_as(local_w2)

    @staticmethod
    def backward(
        ctx: Any,
        grad_local_w1w3: Tensor,
        grad_local_w2: Tensor,
    ) -> tuple[Tensor, Tensor, None, None, None]:
        # Standard GMM allocated and returned these two local [2B] gradients.
        # MoonEP may stage them into its private symmetric reduction slot here;
        # that storage rule never enters GroupedLinear or the GMM op schema.
        home_grads = ctx.invocation._complete_weight_gradients((grad_local_w1w3, grad_local_w2))
        return (
            home_grads[0].reshape(ctx.home_shapes[0]),
            home_grads[1].reshape(ctx.home_shapes[1]),
            None,
            None,
            None,
        )


class MoonEPDispatcher(GenericDispatcher):
    def __init__(self, *, runtime: MoonEPRuntime, layer_id: int) -> None:
        self.runtime = runtime
        self.layer_id = layer_id
        self._next_reduction_slot = 0

    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        layer_state: object | None,
        dispatch_async: bool = False,
        combine_async: bool = False,
        decoding: bool = False,
    ) -> StageState:
        del layer_state, decoding
        return {
            "hidden_states": hidden_states,
            "topk_ids": topk_ids.to(torch.int32).contiguous(),
            "topk_weights": topk_weights,
            # Router histogram is reused; no second bincount(topk_ids).
            "tokens_per_expert": tokens_per_expert.to(torch.int32).contiguous(),
            "_dispatch_async": dispatch_async,
            "_combine_async": combine_async,
        }

    def dispatch(
        self,
        state: StageState,
    ) -> StageState:
        reduction_slot = self._next_reduction_slot
        self._next_reduction_slot = (reduction_slot + 1) % self.runtime.max_inflight
        invocation = create_moonep_invocation(
            self.runtime,
            self.layer_id,
            reduction_slot=reduction_slot,
        )
        hidden_states, weights = invocation.dispatch(
            state["hidden_states"],
            state["topk_ids"],
            # Keep the differentiable cast used by MoonEP's fused route-scaled combine.
            state["topk_weights"].to(torch.float32).contiguous(),
            state["tokens_per_expert"],
            async_op=state["_dispatch_async"],
        )
        return {
            "hidden_states": hidden_states,
            "topk_weights": weights,
            "_invocation": invocation,
            "_combine_async": state["_combine_async"],
        }

    def dispatch_postprocess(
        self,
        state: StageState,
    ) -> StageState:
        invocation = cast(_MoonEPInvocation, state["_invocation"])
        hidden_states, local_counts, weight_layout = invocation.materialize_expert_layout()
        return {
            "hidden_states": hidden_states,
            "tokens_per_expert": local_counts,  # [2B], device resident
            "expert_weight_layout": weight_layout,
            "topk_weights": state["topk_weights"],
            "_invocation": invocation,
            "_combine_async": state["_combine_async"],
        }

    def combine_preprocess(
        self,
        state: StageState,
        expert_output: Tensor,
    ) -> StageState:
        invocation = cast(_MoonEPInvocation, state["_invocation"])
        return {
            "hidden_states": invocation.combine_preprocess(
                expert_output,
                async_op=state["_combine_async"],
            ),
            "topk_weights": state["topk_weights"],
            "_invocation": invocation,
            "_combine_async": state["_combine_async"],
        }

    def combine(
        self,
        state: StageState,
    ) -> StageState:
        invocation = cast(_MoonEPInvocation, state["_invocation"])
        return {
            "hidden_states": invocation.combine(
                state["hidden_states"],
                state["topk_weights"],
                async_op=state["_combine_async"],
            ),
            "_invocation": invocation,
            "_combine_async": state["_combine_async"],
        }

    def combine_postprocess(
        self,
        state: StageState,
    ) -> Tensor:
        invocation = cast(_MoonEPInvocation, state["_invocation"])
        output = invocation.combine_postprocess(
            state["hidden_states"],
            async_op=state["_combine_async"],
        )
        if not torch.is_grad_enabled():
            invocation.finish_forward_only()
        return output


# =============================================================================
# 4. [ULTRAEP ADAPTER] model runtime + DeepEP decorator + ordering nodes
# =============================================================================


class _UltraEPGradReduceJoin(torch.autograd.Function):
    """Forward 位于 attention 前；backward 在 FSDP completion boundary 前 join。"""

    @staticmethod
    def forward(
        ctx: Any,
        layer_input: Tensor,
        runtime: UltraEPRuntime,
        layer_id: int,
        virtual_layer_id: int,
    ) -> Tensor:
        ctx.runtime = runtime
        ctx.layer_id = layer_id
        ctx.virtual_layer_id = virtual_layer_id
        return layer_input

    @staticmethod
    def backward(ctx: Any, grad_input: Tensor) -> tuple[Tensor, None, None, None]:
        ctx.runtime._finish_grad_reduce(ctx.layer_id, ctx.virtual_layer_id)
        return grad_input, None, None, None


class _UltraEPGradReduceStart(torch.autograd.Function):
    """Forward 位于 dispatch 前；backward 在 expert/dispatch backward 后启动 reduce。"""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: Tensor,
        runtime: UltraEPRuntime,
        layer_id: int,
        virtual_layer_id: int,
    ) -> Tensor:
        ctx.runtime = runtime
        ctx.layer_id = layer_id
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states

    @staticmethod
    def backward(ctx: Any, grad_hidden: Tensor) -> tuple[Tensor, None, None, None]:
        ctx.runtime._start_grad_reduce(ctx.layer_id, ctx.virtual_layer_id)
        return grad_hidden, None, None, None


class _UltraEPWeightSyncForBackward(torch.autograd.Function):
    """Forward 位于 expert output 后；backward 在 expert DGrad 前 replay replicas。"""

    @staticmethod
    def forward(
        ctx: Any,
        expert_output: Tensor,
        runtime: UltraEPRuntime,
        virtual_layer_id: int,
    ) -> Tensor:
        ctx.runtime = runtime
        ctx.virtual_layer_id = virtual_layer_id
        return expert_output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        ctx.runtime._replay_weights_for_backward(ctx.virtual_layer_id)
        return grad_output, None, None


class UltraEPRuntime:
    """External Manager、FP32 staging 与 FSDP lifecycle 的 model-scoped owner。"""

    def __init__(
        self,
        *,
        num_experts: int,
        replica_slots_per_rank: int,
        ep_group: Any,
    ) -> None:
        # Validation consumes MoEConfig once. Runtime methods use only these
        # scalars plus the projection pairs registered below.
        self.num_experts = num_experts
        self.replica_slots_per_rank = replica_slots_per_rank
        self.ep_group = ep_group
        self.ep_size = ep_group.size()
        self._closed = False
        self._manager: Any | None = None
        self._layers: list[tuple[str, tuple[nn.Module, nn.Module]]] = []
        self._master_grad_staging: dict[int, ProjectionPair] = {}
        self._active_layer_calls: set[int] = set()
        self._grad_reduce_events: dict[int, object] = {}

    @property
    def num_physical_experts(self) -> int:
        # DeepEP global IDs are rank blocks [B masters, R replicas]; master IDs are
        # interleaved with replica gaps and are not the logical range [0, E).
        return self.num_experts + self.ep_size * self.replica_slots_per_rank

    def bind_dispatcher(
        self,
        *,
        layer_fqn: str,
        projections: tuple[nn.Module, nn.Module],
    ) -> GenericDispatcher:
        if any(registered_fqn == layer_fqn for registered_fqn, _ in self._layers):
            raise ValueError(f"duplicate UltraEP routed layer: {layer_fqn}")
        layer_id = len(self._layers)
        self._layers.append((layer_fqn, projections))
        inner = build_deepep_dispatcher(
            n_routed_experts=self.num_physical_experts,
            ep_group=self.ep_group,
        )
        dispatcher = UltraEPDispatcher(
            runtime=self,
            # UltraEP virtual IDs encode this stable physical decoder ordinal.
            layer_id=layer_id,
            inner=inner,
        )
        return dispatcher

    def validate_before_fsdp(self, *, fsdp_config: Any) -> None:
        validate_ultraep_fsdp_config(fsdp_config)
        validate_two_segment_gmm_supports_registered_projections([projections for _, projections in self._layers])

    def install_after_fsdp(self, *, fsdp_root: nn.Module) -> None:
        # Like MoonEP landing, storage binding may inspect FSDP-owned state from
        # this traversal root. The runtime never retains the root itself.
        self._manager = create_external_ultraep_manager(
            num_experts=self.num_experts,
            replica_slots_per_rank=self.replica_slots_per_rank,
            ep_group=self.ep_group,
            explicitly_destroy=True,
        )
        for layer_id, (layer_fqn, projections) in enumerate(self._layers):
            self._master_grad_staging[layer_id] = bind_ultraep_layer_storage(
                manager=self._manager,
                fsdp_root=fsdp_root,
                layer_id=layer_id,
                layer_fqn=layer_fqn,
                projections=projections,
            )

    def _allocate_virtual_layer_id(self, layer_id: int) -> int:
        # A virtual placement ID does not isolate UltraEP v1's shared replica tensors.
        if layer_id in self._active_layer_calls:
            raise RuntimeError("UltraEP v1 permits one active call per physical layer")
        manager = cast(Any, self._manager)
        virtual_layer_id = manager.allocate_microbatch_slot(layer_id)
        self._active_layer_calls.add(layer_id)
        return virtual_layer_id

    def _prepare_dispatch(
        self,
        layer_id: int,
        virtual_layer_id: int,
        logical_ids: Tensor,
    ) -> tuple[Tensor, object]:
        """Hide Manager placement, registered projections and weight sync。"""
        manager = cast(Any, self._manager)
        manager.update_placement_sparse(virtual_layer_id, logical_ids)
        refresh_external_master_pointers(
            manager,
            layer_id,
            self._layers[layer_id][1],
        )
        weight_sync_event = manager.weight_sync(virtual_layer_id, async_finish=True)
        physical_ids = logical_ids.clone()
        manager.reroute_sparse(virtual_layer_id, physical_ids)
        return physical_ids, weight_sync_event

    def _get_expert_weight_layout(
        self,
        layer_id: int,
        virtual_layer_id: int,
    ) -> ExpertWeightLayout:
        external_weights, external_wgrad_outs = get_ultraep_call_storage(
            manager=cast(Any, self._manager),
            layer_id=layer_id,
            virtual_layer_id=virtual_layer_id,
        )
        return ExpertWeightLayout(
            external_weights=external_weights,
            external_wgrad_outs=external_wgrad_outs,
        )

    def _replay_weights_for_backward(self, virtual_layer_id: int) -> None:
        cast(Any, self._manager).weight_sync(virtual_layer_id, async_finish=False)

    def _start_grad_reduce(self, layer_id: int, virtual_layer_id: int) -> None:
        manager = cast(Any, self._manager)
        stage_master_grads_to_fp32(
            self._master_grad_staging[layer_id],
            virtual_layer_id,
            self._layers[layer_id][1],
        )
        self._grad_reduce_events[virtual_layer_id] = manager.grad_reduce(
            virtual_layer_id,
            async_finish=True,
        )

    def _finish_grad_reduce(self, layer_id: int, virtual_layer_id: int) -> None:
        current_stream_wait_event(self._grad_reduce_events.pop(virtual_layer_id))
        restore_fp32_master_grads_to_fsdp(
            self._master_grad_staging[layer_id],
            virtual_layer_id,
            self._layers[layer_id][1],
        )
        self._release_virtual_layer_id(layer_id)

    def _release_virtual_layer_id(self, layer_id: int) -> None:
        self._active_layer_calls.remove(layer_id)

    def close(self) -> None:
        if not self._closed:
            if self._manager is not None:
                self._manager.destroy()
                self._manager = None
            self._layers.clear()
            self._master_grad_staging.clear()
            self._active_layer_calls.clear()
            self._grad_reduce_events.clear()
            self._closed = True


class UltraEPDispatcher(GenericDispatcher):
    """UltraEP control plane around DeepEP's unchanged communication Implementation。"""

    def __init__(
        self,
        *,
        runtime: UltraEPRuntime,
        layer_id: int,
        inner: GenericDispatcher,
    ) -> None:
        self.runtime = runtime
        self.layer_id = layer_id
        self.inner = inner

    def prepare_layer_input(self, layer_input: Tensor) -> tuple[Tensor, object | None]:
        virtual_layer_id = self.runtime._allocate_virtual_layer_id(self.layer_id)
        joined_input = _UltraEPGradReduceJoin.apply(
            layer_input,
            self.runtime,
            self.layer_id,
            virtual_layer_id,
        )
        return joined_input, virtual_layer_id

    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        layer_state: object | None,
        dispatch_async: bool = False,
        combine_async: bool = False,
        decoding: bool = False,
    ) -> StageState:
        virtual_layer_id = cast(int, layer_state)
        # Router counts remain the common phase-1 contract; UltraEP's real API
        # derives placement loads from IDs and therefore does not consume them.
        physical_ids, weight_sync_event = self.runtime._prepare_dispatch(
            self.layer_id,
            virtual_layer_id,
            topk_ids,
        )
        dispatch_input = _UltraEPGradReduceStart.apply(
            hidden_states,
            self.runtime,
            self.layer_id,
            virtual_layer_id,
        )
        inner_pre = self.inner.dispatch_preprocess(
            hidden_states=dispatch_input,
            topk_ids=physical_ids,
            topk_weights=topk_weights,
            # The inner DeepEP Adapter currently derives physical counts from IDs.
            # It accepts the required source-count argument but does not reinterpret it.
            tokens_per_expert=tokens_per_expert,
            layer_state=None,
            dispatch_async=dispatch_async,
            combine_async=combine_async,
            decoding=decoding,
        )
        return {
            "inner": inner_pre,
            "_virtual_layer_id": virtual_layer_id,
            "_weight_sync_event": weight_sync_event,
        }

    def dispatch(
        self,
        state: StageState,
    ) -> StageState:
        return {
            "inner": self.inner.dispatch(state["inner"]),
            "_virtual_layer_id": state["_virtual_layer_id"],
            "_weight_sync_event": state["_weight_sync_event"],
        }

    def dispatch_postprocess(
        self,
        state: StageState,
    ) -> StageState:
        inner_post = self.inner.dispatch_postprocess(state["inner"])
        # Device-side dependency only; do not synchronize the host.
        current_stream_wait_event(state["_weight_sync_event"])
        return {
            "hidden_states": inner_post["hidden_states"],
            # DeepEP local order is fixed to [B master groups, R replica groups].
            "tokens_per_expert": inner_post["tokens_per_expert"],
            "expert_weight_layout": self.runtime._get_expert_weight_layout(
                self.layer_id,
                state["_virtual_layer_id"],
            ),
            "inner": inner_post,
            "_virtual_layer_id": state["_virtual_layer_id"],
        }

    def combine_preprocess(
        self,
        state: StageState,
        expert_output: Tensor,
    ) -> StageState:
        virtual_layer_id = cast(int, state["_virtual_layer_id"])
        replay_edge = _UltraEPWeightSyncForBackward.apply(
            expert_output,
            self.runtime,
            virtual_layer_id,
        )
        inner_pre_combined = self.inner.combine_preprocess(
            state["inner"],
            replay_edge,
        )
        return {
            "inner": inner_pre_combined,
            "_virtual_layer_id": virtual_layer_id,
        }

    def combine(
        self,
        state: StageState,
    ) -> StageState:
        return {
            "inner": self.inner.combine(state["inner"]),
            "_virtual_layer_id": state["_virtual_layer_id"],
        }

    def combine_postprocess(
        self,
        state: StageState,
    ) -> Tensor:
        output = self.inner.combine_postprocess(state["inner"])
        if not torch.is_grad_enabled():
            self.runtime._release_virtual_layer_id(self.layer_id)
        return output


# =============================================================================
# 5. [CHANGED EXPERT COMPUTE] backend-neutral GroupedLinear Interface
# =============================================================================


class GroupedLinear(nn.Module):
    """Trainable parameter owner with one counts-based caller Interface.

    This existing BF16 Module calls the standard one-segment op directly. Only
    UltraEP's real two-allocation case enters the two-segment Implementation.
    """

    weight: nn.Parameter
    local_out_features: int
    local_in_features: int

    def forward(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        *,
        trainable_weight: Tensor | None = None,
        external_weight: Tensor | None = None,
        external_wgrad_out: Tensor | None = None,
    ) -> Tensor:
        if trainable_weight is None:
            trainable_weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
            trainable_weight = trainable_weight.view(
                -1,
                self.local_out_features,
                self.local_in_features,
            )

        if external_weight is None:
            # Ordinary EP and MoonEP use the standard one-segment autograd path.
            # MoonEP captures its natural dW on the differentiable weight alias.
            return group_gemm(hidden_states, trainable_weight, tokens_per_expert)

        # The only real two-allocation caller is UltraEP. Its external weight
        # and WGrad tensors may have different expert strides and dtypes.
        assert external_wgrad_out is not None
        return _TwoSegmentGroupedLinear.apply(
            hidden_states,
            trainable_weight,
            external_weight,
            external_wgrad_out,
            tokens_per_expert,
        )


class TileWiseFloat8GroupedLinear(nn.Module):
    """Existing FP8 Module implementing the same caller Interface directly.

    Ordinary EP may receive an FSDP-prequantized weight. MoonEP deliberately
    supplies a BF16 local [2B] alias; this Adapter dynamically quantizes it and
    natural autograd still returns BF16 dW to _ExpertWeightAutograd.
    """

    weight: nn.Parameter

    def forward(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        *,
        trainable_weight: Tensor | None = None,
        external_weight: Tensor | None = None,
        external_wgrad_out: Tensor | None = None,
    ) -> Tensor:
        # UltraEP FP8 is rejected during model validation. Keeping its unsupported
        # storage operands out of the op preserves the existing AdaptiveGEMM ABI.
        if external_weight is not None or external_wgrad_out is not None:
            raise RuntimeError("UltraEP FP8 is not supported")
        if trainable_weight is None:
            trainable_weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return tilewise_fp8_group_gemm(
            hidden_states,
            trainable_weight,
            tokens_per_expert,
        )


class MoEBlock(nn.Module):
    """Owns fused projections; no Dispatcher/backend names enter this Module."""

    def __init__(
        self,
        fused_w1w3: nn.Module,  # BF16 or existing TileWise FP8 Adapter
        fused_w2: nn.Module,
        moe_act: nn.Module,
    ) -> None:
        super().__init__()
        self.fused_w1w3 = fused_w1w3
        self.fused_w2 = fused_w2
        self.moe_act = moe_act

    def forward(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        *,
        weight_layout: ExpertWeightLayout,
    ) -> Tensor:
        trainable_w1w3 = None if weight_layout.trainable_weights is None else weight_layout.trainable_weights[0]
        trainable_w2 = None if weight_layout.trainable_weights is None else weight_layout.trainable_weights[1]
        external_w1w3 = None if weight_layout.external_weights is None else weight_layout.external_weights[0]
        external_w2 = None if weight_layout.external_weights is None else weight_layout.external_weights[1]
        external_dw1w3 = None if weight_layout.external_wgrad_outs is None else weight_layout.external_wgrad_outs[0]
        external_dw2 = None if weight_layout.external_wgrad_outs is None else weight_layout.external_wgrad_outs[1]

        gate_up = self.fused_w1w3(
            hidden_states,
            tokens_per_expert,
            trainable_weight=trainable_w1w3,
            external_weight=external_w1w3,
            external_wgrad_out=external_dw1w3,
        )
        return self.fused_w2(
            self.moe_act(gate_up),
            tokens_per_expert,
            trainable_weight=trainable_w2,
            external_weight=external_w2,
            external_wgrad_out=external_dw2,
        )


# =============================================================================
# 6. [CLIENT] Decoder 单 microbatch 与 Domino 的主要流程
# =============================================================================


class MoEDecoderLayer(nn.Module):
    dispatcher: GenericDispatcher
    experts: MoEBlock
    n_shared_experts: int

    def _forward(
        self,
        layer_input: Tensor,
        seq_ctx: Any,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # The only hook outside the existing six stages. For all non-UltraEP Adapters
        # this is an identity; opaque state is handed directly to phase 1.
        layer_input, layer_state = self.dispatcher.prepare_layer_input(layer_input)

        residual, routed_hidden, router = self._pre_moe_forward(
            hidden_states=layer_input,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
        )
        origin_shape = routed_hidden.shape

        state = self.dispatcher.dispatch_preprocess(
            hidden_states=routed_hidden.view(-1, routed_hidden.shape[-1]),
            topk_ids=router["topk_ids"],
            topk_weights=router["topk_weights"],
            tokens_per_expert=router["tokens_per_expert"],
            layer_state=layer_state,
            dispatch_async=False,
            combine_async=True,
            decoding=False,
        )
        state = self.dispatcher.dispatch(state)
        state = self.dispatcher.dispatch_postprocess(state)

        expert_output = self.experts(
            state["hidden_states"],
            state["tokens_per_expert"],
            weight_layout=state["expert_weight_layout"],
        )
        state = self.dispatcher.combine_preprocess(state, expert_output)
        state = self.dispatcher.combine(state)

        # Existing overlap stays: routed combine runs while shared experts compute.
        shared_output = self._shared_experts_forward(routed_hidden) if self.n_shared_experts > 0 else None
        routed_output = self.dispatcher.combine_postprocess(state).view(*origin_shape)
        output = self._post_moe_forward(
            combined_hidden_states=routed_output,
            residual=residual,
            shared_experts_out=shared_output,
        )
        # Public return remains logical; UltraEP physical IDs stay inside its Dispatcher.
        return output, router["logits"], router["router_weights"], router["topk_ids"]

    def _micro_batch_forward(
        self,
        layer_inputs: Sequence[Tensor],
        seq_ctxs: Sequence[Any],
        position_embeddings: Sequence[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, ...]:
        # This common scheduler is enabled only for backends whose validation declares
        # intra-layer concurrency. MoonEP uses its static ordinal ring; UltraEP v1 rejects it.
        residuals: list[Tensor] = []
        routed_hiddens: list[Tensor] = []
        routers: list[RouterResults] = []
        states: list[StageState] = []

        # Attention + logical router + phase 1.
        for layer_input, seq_ctx, pos_emb in zip(layer_inputs, seq_ctxs, position_embeddings):
            layer_input, layer_state = self.dispatcher.prepare_layer_input(layer_input)
            residual, routed_hidden, router = self._pre_moe_forward(
                hidden_states=layer_input,
                seq_ctx=seq_ctx,
                position_embeddings=pos_emb,
            )
            state = self.dispatcher.dispatch_preprocess(
                hidden_states=routed_hidden.view(-1, routed_hidden.shape[-1]),
                topk_ids=router["topk_ids"],
                topk_weights=router["topk_weights"],
                tokens_per_expert=router["tokens_per_expert"],
                layer_state=layer_state,
                dispatch_async=True,
                combine_async=True,
                decoding=False,
            )
            residuals.append(residual)
            routed_hiddens.append(routed_hidden)
            routers.append(router)
            states.append(state)

        # Preserve xtuner_ep_domino.md's Loop B: phases 2-4 stay consecutive
        # for each microbatch.  D1 can overlap E0/Cpre0 on another stream.
        for index, state in enumerate(states):
            state = self.dispatcher.dispatch(state)
            state = self.dispatcher.dispatch_postprocess(state)
            # Counts and weight layout belong to this microbatch's autograd
            # graph; backend metadata and MoonEP reduction slots stay private.
            expert_output = self.experts(
                state["hidden_states"],
                state["tokens_per_expert"],
                weight_layout=state["expert_weight_layout"],
            )
            states[index] = self.dispatcher.combine_preprocess(state, expert_output)

        # Phase 5 is launched for all microbatches before shared expert compute.
        for index, state in enumerate(states):
            states[index] = self.dispatcher.combine(state)

        shared_outputs = [
            self._shared_experts_forward(hidden) if self.n_shared_experts > 0 else None for hidden in routed_hiddens
        ]

        outputs: list[Tensor] = []
        for state, residual, routed_hidden, shared_output in zip(
            states,
            residuals,
            routed_hiddens,
            shared_outputs,
        ):
            routed_output = self.dispatcher.combine_postprocess(state).view_as(routed_hidden)
            outputs.append(
                self._post_moe_forward(
                    combined_hidden_states=routed_output,
                    residual=residual,
                    shared_experts_out=shared_output,
                )
            )

        return tuple(
            outputs
            + [router["logits"] for router in routers]
            + [router["router_weights"] for router in routers]
            + [router["topk_ids"] for router in routers]
        )

    def _pre_moe_forward(
        self,
        *,
        hidden_states: Tensor,
        seq_ctx: Any,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, RouterResults]:
        raise NotImplementedError

    def _shared_experts_forward(self, hidden_states: Tensor) -> Tensor:
        raise NotImplementedError

    def _post_moe_forward(
        self,
        *,
        combined_hidden_states: Tensor,
        residual: Tensor,
        shared_experts_out: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError


# =============================================================================
# 7. [CLIENT] Config、model build、FSDP install 与 teardown
# =============================================================================


class MoEConfig:
    dispatcher: DispatcherName
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    expert_compute_dtype: Literal["bf16", "fp8"]
    expert_tp_size: int
    data_parallel_size: int
    intra_layer_micro_batch: int
    moonep_num_sms: int
    moonep_staging_reference: bool
    mtp_config: Any | None
    moe_bias: bool
    ultraep_cfg: Any | None
    moonep_cfg: Any | None


def build_ep_runtime(
    *,
    config: MoEConfig,
    ep_group: Any,
) -> MoonEPRuntime | UltraEPRuntime | None:
    """Validate the full config once, then retain only runtime-owned facts。"""
    if config.dispatcher == "moonep":
        validate_moonep_python_contract(config, ep_group)
        return MoonEPRuntime(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            max_inflight=config.intra_layer_micro_batch,
            num_sms=config.moonep_num_sms,
            staging_reference=config.moonep_staging_reference,
            ep_group=ep_group,
        )
    if config.dispatcher == "ultraep":
        validate_ultraep_python_contract(config, ep_group)
        return UltraEPRuntime(
            num_experts=config.n_routed_experts,
            replica_slots_per_rank=config.ultraep_cfg.replica_slots_per_rank,
            ep_group=ep_group,
        )
    return None


class MoEModel(nn.Module):
    def __init__(self, config: MoEConfig, ep_group: Any) -> None:
        super().__init__()
        self.config = config
        self.ep_group = ep_group
        self._ep_runtime = build_ep_runtime(config=config, ep_group=ep_group)

        # Every constructed routed-expert layer gets a per-layer Adapter from the same runtime.
        # UltraEP v1 rejects MTP during resource-free runtime validation before this build.
        self.layers = build_decoder_layers(
            config=config,
            # This assembly lambda is the last boundary that sees MoEBlock. It
            # also keeps expert compute dtype out of dynamic transport runtimes.
            dispatcher_builder=lambda layer_fqn, experts: (
                self._ep_runtime.bind_dispatcher(
                    layer_fqn=layer_fqn,
                    projections=(experts.fused_w1w3, experts.fused_w2),
                )
                if self._ep_runtime is not None
                else build_existing_dispatcher(
                    dispatcher=config.dispatcher,
                    n_routed_experts=config.n_routed_experts,
                    ep_group=ep_group,
                    transport_dtype=config.expert_compute_dtype,
                )
            ),
        )

    def fully_shard(self, fsdp_config: Any) -> MoEModel:
        # Critical ordering: reject unsupported combinations before mutating parameters.
        if self._ep_runtime is not None:
            self._ep_runtime.validate_before_fsdp(fsdp_config=fsdp_config)

        existing_fully_shard_model(self, fsdp_config)

        if self._ep_runtime is not None:
            self._ep_runtime.install_after_fsdp(fsdp_root=self)
        return self

    def close_ep_runtime(self) -> None:
        if self._ep_runtime is not None:
            self._ep_runtime.close()


class TrainEngine:
    model: MoEModel

    def close(self) -> None:
        self.model.close_ep_runtime()
        close_other_engine_resources(self)


def trainer_normal_teardown(trainer: Any) -> None:
    """Normal coordinated caller flow; destructors never enter collectives。"""
    wait_for_pending_async_hf_exports(trainer)
    wait_for_pending_async_checkpoints(trainer)
    torch.distributed.barrier()
    trainer.engine.close()
    # The CLI destroys process groups only after all ranks return from engine.close().


# =============================================================================
# 8. [KERNEL CONTRACT] BF16 two-segment Implementation
# =============================================================================


class _TwoSegmentGroupedLinear(torch.autograd.Function):
    """UltraEP autograd Adapter around stride-aware BF16 custom ops.

    Counts are the Interface. Triton tile metadata is private to the two ops;
    CUTLASS and FP8 never need to understand it.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        trainable_weight: Tensor,
        external_weight: Tensor,
        external_wgrad_out: Tensor,
        tokens_per_expert: Tensor,
    ) -> Tensor:
        # Runtime-owned replicas may be replayed before DGrad.  Keeping this as a
        # plain ctx attribute avoids a saved-tensor version check on refreshed data.
        ctx.external_weight = external_weight
        ctx.external_wgrad_out = external_wgrad_out
        ctx.save_for_backward(x, trainable_weight, tokens_per_expert)
        return two_segment_group_gemm_forward_op(
            x,
            trainable_weight,
            external_weight,
            tokens_per_expert,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        x, trainable_weight, tokens_per_expert = ctx.saved_tensors
        trainable_dw = torch.empty_like(trainable_weight)

        dx = two_segment_group_gemm_backward_out_op(
            grad_output,
            x,
            trainable_weight,
            ctx.external_weight,
            trainable_dw,
            ctx.external_wgrad_out,
            tokens_per_expert,
        )
        # external_weight is runtime-owned: its dW is only the explicit side output.
        return dx, trainable_dw, None, None, None


# =============================================================================
# 9. Omitted production implementations / external boundaries
# =============================================================================


def validate_moonep_python_contract(config: Any, ep_group: Any) -> None: ...


def create_moonep_invocation(
    runtime: MoonEPRuntime,
    layer_id: int,
    *,
    reduction_slot: int,
) -> _MoonEPInvocation: ...


def materialize_moonep_local_tensors(
    invocation: _MoonEPInvocation,
) -> tuple[Tensor, Tensor, ProjectionPair]:
    """Return local activations/counts and weights wrapped by _ExpertWeightAutograd。"""
    ...


def validate_moonep_fsdp_config(fsdp_config: Any) -> None: ...


def install_moonep_vmm_and_fsdp_landing(
    runtime: MoonEPRuntime,
    fsdp_root: nn.Module,
) -> None: ...


def destroy_moonep_buffers_vmm_and_landing(runtime: MoonEPRuntime) -> None: ...


def create_external_ultraep_manager(
    *,
    num_experts: int,
    replica_slots_per_rank: int,
    ep_group: Any,
    explicitly_destroy: bool,
) -> Any: ...


def bind_ultraep_layer_storage(
    *,
    manager: Any,
    fsdp_root: nn.Module,
    layer_id: int,
    layer_fqn: str,
    projections: tuple[nn.Module, nn.Module],
) -> ProjectionPair:
    """Register two projections and return this layer's FP32 master staging。"""
    ...


def get_ultraep_call_storage(
    *,
    manager: Any,
    layer_id: int,
    virtual_layer_id: int,
) -> tuple[ProjectionPair, ProjectionPair]:
    """Return (external weights, external WGrad targets) for this call only。"""
    ...


def refresh_external_master_pointers(
    manager: Any,
    layer_id: int,
    projections: tuple[nn.Module, nn.Module],
) -> None: ...


def stage_master_grads_to_fp32(
    master_grad_staging: ProjectionPair,
    virtual_layer_id: int,
    projections: tuple[nn.Module, nn.Module],
) -> None: ...


def restore_fp32_master_grads_to_fsdp(
    master_grad_staging: ProjectionPair,
    virtual_layer_id: int,
    projections: tuple[nn.Module, nn.Module],
) -> None: ...


def current_stream_wait_event(event: object) -> None: ...


def validate_ultraep_python_contract(config: Any, ep_group: Any) -> None:
    unsupported = (
        ep_group.size() <= 1
        or config.ultraep_cfg is None
        or config.expert_compute_dtype != "bf16"
        or config.expert_tp_size != 1
        or config.data_parallel_size != 1
        or config.intra_layer_micro_batch != 1
        or config.mtp_config is not None
        or config.moe_bias
    )
    if unsupported:
        raise ValueError("UltraEP v1 requires EP>1, BF16, TP1, DP1, microbatch1, bias-free experts, and no MTP")
    if config.ultraep_cfg.replica_slots_per_rank <= 0:
        raise ValueError("UltraEP replica_slots_per_rank must be positive")


def validate_ultraep_fsdp_config(fsdp_config: Any) -> None:
    """UltraEP v1 keeps the PR's deliberately narrow, proven capability boundary。"""
    if fsdp_config.recompute_ratio > 0:
        raise ValueError("UltraEP v1 does not support activation recompute")


def validate_two_segment_gmm_supports_registered_projections(
    projections: Sequence[tuple[nn.Module, nn.Module]],
) -> None: ...


def build_deepep_dispatcher(*, n_routed_experts: int, ep_group: Any) -> GenericDispatcher: ...


def group_gemm(
    x: Tensor,
    weight: Tensor,
    tokens_per_expert: Tensor,
) -> Tensor:
    """Selected standard one-segment Adapter with natural autograd WGrad.

    Triton and CUTLASS keep their existing allocation-return contract. TileWise
    FP8 stays behind its own GroupedLinear Implementation because its raw op also
    consumes quantization scales.
    """
    ...


def tilewise_fp8_group_gemm(
    x: Tensor,
    weight: Tensor,
    tokens_per_expert: Tensor,
) -> Tensor:
    """Hide dynamic block quant/scales and return natural BF16 dW.

    ``weight`` may be an ordinary prequantized FSDP value or MoonEP's BF16
    local [2B] override. UltraEP external segments are rejected at build time.
    """
    ...


def two_segment_group_gemm_forward_op(
    x: Tensor,
    trainable_weight: Tensor,
    external_weight: Tensor,
    counts: Tensor,
) -> Tensor:
    """Stride-aware BF16 forward; all tile metadata is op-private。"""
    ...


def two_segment_group_gemm_backward_out_op(
    grad_output: Tensor,
    x: Tensor,
    trainable_weight: Tensor,
    external_weight: Tensor,
    trainable_dw_out: Tensor,
    external_dw_out: Tensor,
    counts: Tensor,
) -> Tensor:
    """Return DGrad and cover both WGrad targets declared in ``mutates_args``。"""
    ...


def build_existing_dispatcher(
    *,
    dispatcher: DispatcherName,
    n_routed_experts: int,
    ep_group: Any,
    transport_dtype: Literal["bf16", "fp8"],
) -> GenericDispatcher: ...


def build_decoder_layers(*, config: MoEConfig, dispatcher_builder: Any) -> nn.ModuleList: ...


def existing_fully_shard_model(model: MoEModel, fsdp_config: Any) -> None: ...


def wait_for_pending_async_hf_exports(engine: TrainEngine) -> None: ...


def wait_for_pending_async_checkpoints(engine: TrainEngine) -> None: ...


def close_other_engine_resources(engine: TrainEngine) -> None: ...


# Public-behavior acceptance surface:
#
# - build -> fully_shard -> forward -> backward -> optimizer.step parity;
# - Router logical IDs/counts never mutate or leak physical semantics;
# - MoonEP / UltraEP completion happens before FSDP post-backward;
# - official R=2 strided replica views work, or config rejects them before FSDP mutation;
# - UltraEP v1 rejects a second same-layer call before shared storage is overwritten;
# - each backend only enables Domino/MTP/reentrant paths declared by its capability boundary;
# - runtimes do not retain the full model/config, and close releases registered projections;
# - ordinary and MoonEP BF16 GMM paths use the same three-argument Interface;
# - MoonEP FP8 dynamically quantizes a BF16 local override and returns natural BF16 dW,
#   or rejects the configuration before resource/FSDP mutation;
# - ordinary FP8/CUTLASS paths still use their natural autograd WGrad;
# - UltraEP v1 rejects FP8 rather than exposing quantization metadata to Dispatcher;
# - DCP/HF state excludes all runtime/private-state tensors and cold-runtime restore works;
# - explicit close permits rebuilding a different runtime in the same process.
