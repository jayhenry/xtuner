"""MoonEP / UltraEP 一致接入方案的结构化伪代码。

这不是可直接替换 XTuner 文件的 patch。它刻意展示：

1. model-scoped runtime Interface；
2. per-layer Dispatcher Adapter 与已有六阶段；
3. per-call invocation façade；
4. storage-neutral expert execution；
5. 单 microbatch、Domino、FSDP install 和 teardown 的主要调用端流程。

省略了真实 backend import、stream 细节、错误文本和现有 Dispatcher 的具体通信代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Sequence, TypedDict, cast

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor


StageResult = dict[str, Any]
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


class ProjectionExecution(NamedTuple):
    """一层 grouped projection 的 storage-neutral tensor contract。

    ``primary_weight=None`` 表示使用 GroupedLinear 自己的 Parameter。
    ``secondary_weight`` 存在时，counts 的 group 顺序固定为 primary 后 secondary。
    """

    primary_weight: Tensor | None = None
    primary_grad_out: Tensor | None = None
    secondary_weight: Tensor | None = None
    secondary_grad_out: Tensor | None = None


class ExpertExecution(NamedTuple):
    gate_up: ProjectionExecution
    down: ProjectionExecution


class ExpertBatch(NamedTuple):
    hidden_states: Tensor
    tokens_per_expert: Tensor  # local physical compute groups, not Router [E] counts
    execution: ExpertExecution | None


# =============================================================================
# 2. [CHANGED BASE] 既有六阶段 + per-call façade
# =============================================================================


class GenericDispatcher(ABC):
    """保留现有六阶段 Interface。

    新增的 ``begin_invocation`` 是 pre-attention 调用生命周期入口；默认是 identity。
    ``invocation_state`` 只在 eager control plane 传递，后续阶段可从 pre result 取得。
    """

    def begin_invocation(self, layer_input: Tensor) -> RoutedExpertInvocation:
        return RoutedExpertInvocation(
            dispatcher=self,
            layer_input=layer_input,
            backend_state=None,
        )

    @abstractmethod
    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        invocation_state: object | None,
        async_op: bool = False,
    ) -> StageResult:
        """阶段 1；tokens_per_expert 是 Router-owned logical [E] counts。"""

    @abstractmethod
    def dispatch(
        self,
        *,
        pre_dispatched: StageResult,
        topk_weights: Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        """阶段 2。"""

    @abstractmethod
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        async_op: bool = False,
    ) -> StageResult:
        """阶段 3；必须固定返回 expert_execution key，普通路径值为 None。"""

    @abstractmethod
    def combine_preprocess(
        self,
        *,
        hidden_states: Tensor,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        """阶段 4。"""

    @abstractmethod
    def combine(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        pre_combined: StageResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        """阶段 5。"""

    @abstractmethod
    def combine_postprocess(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        pre_combined: StageResult,
        combined: StageResult,
        async_op: bool = False,
    ) -> StageResult:
        """阶段 6。"""


class RoutedExpertInvocation:
    """一次 routed-expert call 的有状态 façade。

    它只编排现有 Dispatcher 六阶段，不实现 transport。调用顺序是业务 invariant，
    所以伪代码不为每一步增加浅层 defensive state machine。
    """

    def __init__(
        self,
        *,
        dispatcher: GenericDispatcher,
        layer_input: Tensor,
        backend_state: object | None,
    ) -> None:
        self.dispatcher = dispatcher
        self.layer_input = layer_input
        self.backend_state = backend_state
        self.topk_weights: Tensor
        self.pre_dispatched: StageResult
        self.dispatched: StageResult
        self.post_dispatched: StageResult
        self.pre_combined: StageResult
        self.combined: StageResult

    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        async_op: bool = False,
    ) -> None:
        self.topk_weights = topk_weights
        self.pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            tokens_per_expert=tokens_per_expert,
            invocation_state=self.backend_state,
            async_op=async_op,
        )

    def dispatch(self, *, async_op: bool = False, decoding: bool = False) -> None:
        self.dispatched = self.dispatcher.dispatch(
            pre_dispatched=self.pre_dispatched,
            topk_weights=self.topk_weights,
            async_op=async_op,
            decoding=decoding,
        )

    def dispatch_postprocess(self, *, async_op: bool = False) -> ExpertBatch:
        self.post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=self.pre_dispatched,
            dispatched=self.dispatched,
            async_op=async_op,
        )
        return ExpertBatch(
            hidden_states=self.post_dispatched["hidden_states"],
            tokens_per_expert=self.post_dispatched["tokens_per_expert"],
            # 所有 Adapter 固定写 key，避免 compile 路径使用 optional-key .get()。
            execution=self.post_dispatched["expert_execution"],
        )

    def combine_preprocess(
        self,
        expert_output: Tensor,
        *,
        async_op: bool = False,
        decoding: bool = False,
    ) -> None:
        self.pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=expert_output,
            pre_dispatched=self.pre_dispatched,
            dispatched=self.dispatched,
            post_dispatched=self.post_dispatched,
            async_op=async_op,
            decoding=decoding,
        )

    def combine(self, *, async_op: bool = False, decoding: bool = False) -> None:
        self.combined = self.dispatcher.combine(
            pre_dispatched=self.pre_dispatched,
            dispatched=self.dispatched,
            post_dispatched=self.post_dispatched,
            pre_combined=self.pre_combined,
            async_op=async_op,
            decoding=decoding,
        )

    def combine_postprocess(self, *, async_op: bool = False) -> Tensor:
        post = self.dispatcher.combine_postprocess(
            pre_dispatched=self.pre_dispatched,
            dispatched=self.dispatched,
            post_dispatched=self.post_dispatched,
            pre_combined=self.pre_combined,
            combined=self.combined,
            async_op=async_op,
        )
        return post["hidden_states"]


# Existing Naive/All2All/DeepEP/AGRS Adapters only need two mechanical changes:
#
# 1. dispatch_preprocess accepts required Router tokens_per_expert and invocation_state;
# 2. dispatch_postprocess always returns expert_execution=None.
#
# Their communication Implementation and remaining five stages stay unchanged.


# =============================================================================
# 3. [NEW COMMON LIFECYCLE] model-scoped runtime Interface
# =============================================================================


class EPExecutionRuntime(ABC):
    """模型级 dynamic-expert resources 的 owner。"""

    @abstractmethod
    def bind_dispatcher(self, *, layer_fqn: str, experts: MoEBlock) -> GenericDispatcher:
        """Meta build 时注册 physical layer；不得创建 per-call state。"""

    @abstractmethod
    def validate_before_fsdp(self, *, model: nn.Module, fsdp_config: Any) -> None:
        """任何 FSDP mutation / CUDA allocation 前集中拒绝 unsupported config。"""

    @abstractmethod
    def install_after_fsdp(self, *, model: nn.Module, fsdp_config: Any) -> None:
        """FSDP parameter identity/layout 明确后安装 backend resources。"""

    @abstractmethod
    def close(self) -> None:
        """幂等；必须在相关 process groups 销毁前协调调用。"""


# =============================================================================
# 4. [MOONEP ADAPTER] 保留 Buffer / VMM / private autograd Implementation
# =============================================================================


class _MoonEPInvocation:
    """MoonEP per-call plan/events/gradient slot；真实逻辑复用当前实现。"""

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

    def materialize_expert_execution(
        self,
    ) -> tuple[Tensor, Tensor, ExpertExecution]:
        """返回 dispatched hidden、local [2B] counts 和两 projection specs。"""
        hidden_states, local_counts, weights, grad_outputs = materialize_moonep_local_tensors(self)
        w1w3, w2 = weights  # Differentiable _ExpertWeightAutograd outputs.
        dw1w3, dw2 = grad_outputs
        return (
            hidden_states,
            local_counts,
            ExpertExecution(
                gate_up=ProjectionExecution(
                    primary_weight=w1w3,
                    primary_grad_out=dw1w3,
                ),
                down=ProjectionExecution(
                    primary_weight=w2,
                    primary_grad_out=dw2,
                ),
            ),
        )

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


class MoonEPRuntime(EPExecutionRuntime):
    """现有 model-scoped MoonEP runtime 的通用 lifecycle Adapter。"""

    def __init__(self, *, config: Any, ep_group: Any) -> None:
        self.config = config
        self.ep_group = ep_group
        self._closed = False
        # Resource-free: lazy import/API metadata validation only.
        validate_moonep_python_contract(config, ep_group)

    def bind_dispatcher(self, *, layer_fqn: str, experts: MoEBlock) -> GenericDispatcher:
        binding = register_moonep_layer(self, layer_fqn, experts)
        return MoonEPDispatcher(runtime=self, layer_binding=binding)

    def begin_invocation(self, layer_binding: object) -> _MoonEPInvocation:
        # Fresh plan state. The real v1 dispatcher selects a deterministic ordinal in its
        # statically sized Domino gradient ring; it is not a dynamic completion-tracked lease.
        # A second independent graph is outside the v1 client contract; scheduler/config
        # boundaries must not construct it, and this hot path adds no host completion tracker.
        return create_moonep_invocation(self, layer_binding)

    def validate_before_fsdp(self, *, model: nn.Module, fsdp_config: Any) -> None:
        validate_moonep_fsdp_config(model, fsdp_config)

    def install_after_fsdp(self, *, model: nn.Module, fsdp_config: Any) -> None:
        install_moonep_vmm_and_fsdp_landing(self, model, fsdp_config)

    def close(self) -> None:
        if not self._closed:
            destroy_moonep_buffers_vmm_and_landing(self)
            self._closed = True


class MoonEPDispatcher(GenericDispatcher):
    def __init__(self, *, runtime: MoonEPRuntime, layer_binding: object) -> None:
        self.runtime = runtime
        self.layer_binding = layer_binding

    def begin_invocation(self, layer_input: Tensor) -> RoutedExpertInvocation:
        state = self.runtime.begin_invocation(self.layer_binding)
        return RoutedExpertInvocation(
            dispatcher=self,
            layer_input=layer_input,  # MoonEP does not need a pre-attention join.
            backend_state=state,
        )

    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        invocation_state: object | None,
        async_op: bool = False,
    ) -> StageResult:
        invocation = cast(_MoonEPInvocation, invocation_state)
        return {
            "hidden_states": hidden_states,
            "topk_ids": topk_ids.to(torch.int32).contiguous(),
            "topk_weights": topk_weights,
            # Router histogram is reused; no second bincount(topk_ids).
            "tokens_per_expert": tokens_per_expert.to(torch.int32).contiguous(),
            "_invocation": invocation,
        }

    def dispatch(
        self,
        *,
        pre_dispatched: StageResult,
        topk_weights: Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        invocation = cast(_MoonEPInvocation, pre_dispatched["_invocation"])
        hidden_states, weights = invocation.dispatch(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"],
            # Keep the differentiable cast used by MoonEP's fused route-scaled combine.
            topk_weights.to(torch.float32).contiguous(),
            pre_dispatched["tokens_per_expert"],
            async_op=async_op,
        )
        return {
            "hidden_states": hidden_states,
            "topk_weights": weights,
        }

    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        async_op: bool = False,
    ) -> StageResult:
        invocation = cast(_MoonEPInvocation, pre_dispatched["_invocation"])
        hidden_states, local_counts, execution = invocation.materialize_expert_execution()
        return {
            "hidden_states": hidden_states,
            "tokens_per_expert": local_counts,  # [2B], device resident
            "expert_execution": execution,
        }

    def combine_preprocess(
        self,
        *,
        hidden_states: Tensor,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        invocation = cast(_MoonEPInvocation, pre_dispatched["_invocation"])
        return {
            "hidden_states": invocation.combine_preprocess(hidden_states, async_op=async_op),
        }

    def combine(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        pre_combined: StageResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        invocation = cast(_MoonEPInvocation, pre_dispatched["_invocation"])
        return {
            "hidden_states": invocation.combine(
                pre_combined["hidden_states"],
                dispatched["topk_weights"],
                async_op=async_op,
            ),
        }

    def combine_postprocess(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        pre_combined: StageResult,
        combined: StageResult,
        async_op: bool = False,
    ) -> StageResult:
        invocation = cast(_MoonEPInvocation, pre_dispatched["_invocation"])
        output = invocation.combine_postprocess(combined["hidden_states"], async_op=async_op)
        if not torch.is_grad_enabled():
            invocation.finish_forward_only()
        return {"hidden_states": output}


# =============================================================================
# 5. [ULTRAEP ADAPTER] model runtime + DeepEP decorator + ordering nodes
# =============================================================================


@dataclass(frozen=True)
class UltraEPLayerBinding:
    layer_fqn: str
    experts: MoEBlock
    external_layer_handle: object | None


class UltraEPManagerAdapter:
    """External UltraEP API、本地 staging 和显式销毁的唯一 Locality 边界。"""

    def __init__(self, *, config: Any, ep_group: Any) -> None:
        self.external_manager = create_external_ultraep_manager(
            config=config,
            ep_group=ep_group,
            explicitly_destroy=True,
        )
        self._active_layer_calls: set[str] = set()

    def register_layer(self, layer_fqn: str, experts: MoEBlock) -> UltraEPLayerBinding:
        handle = self.external_manager.register_layer(layer_fqn)
        return UltraEPLayerBinding(layer_fqn, experts, handle)

    def lease_slot(self, binding: UltraEPLayerBinding) -> object:
        # UltraEP v1 has shared replica weight/grad storage. A virtual placement slot alone
        # cannot isolate a second call of the same physical layer.
        if binding.layer_fqn in self._active_layer_calls:
            raise RuntimeError("UltraEP v1 permits one active invocation per physical layer")
        slot = self.external_manager.allocate_microbatch_slot(binding.external_layer_handle)
        self._active_layer_calls.add(binding.layer_fqn)
        return slot

    def refresh_master_pointers(self, binding: UltraEPLayerBinding) -> None:
        # Isolates external movable-pointer knowledge after each FSDP unshard.
        refresh_external_master_pointers(self.external_manager, binding.experts)

    def update_placement(
        self,
        *,
        binding: UltraEPLayerBinding,
        slot: object,
        logical_ids: Tensor,
        logical_counts: Tensor,
    ) -> object:
        self.external_manager.update_placement_sparse(
            binding.external_layer_handle,
            slot,
            logical_ids,
            logical_counts,
        )
        # The external API mutates placement tables owned by this virtual slot.
        return slot

    def weight_sync(self, *, slot: object, placement: object, async_finish: bool) -> object:
        return self.external_manager.weight_sync(slot, placement, async_finish=async_finish)

    def reroute(self, *, slot: object, placement: object, logical_ids: Tensor) -> Tensor:
        # The Router-owned logical tensor remains untouched.
        physical_ids = logical_ids.clone()
        self.external_manager.reroute_sparse(slot, placement, physical_ids)
        return physical_ids

    def replica_execution(self, *, slot: object) -> ExpertExecution:
        replica_w1, replica_w2 = self.external_manager.replica_weights(slot)
        replica_dw1, replica_dw2 = self.external_manager.replica_grad_buffers(slot)
        return ExpertExecution(
            gate_up=ProjectionExecution(
                primary_weight=None,  # GroupedLinear master Parameter [B]
                primary_grad_out=None,
                secondary_weight=replica_w1,  # official strided [R, ...] view
                secondary_grad_out=replica_dw1,
            ),
            down=ProjectionExecution(
                primary_weight=None,
                primary_grad_out=None,
                secondary_weight=replica_w2,
                secondary_grad_out=replica_dw2,
            ),
        )

    def replay_weights(self, *, slot: object, placement: object) -> None:
        self.external_manager.weight_sync(slot, placement, async_finish=False)

    def start_grad_reduce(
        self,
        *,
        binding: UltraEPLayerBinding,
        slot: object,
        placement: object,
    ) -> object:
        stage_master_grads_to_fp32(binding.experts)
        return self.external_manager.grad_reduce(slot, placement, async_finish=True)

    def finish_grad_reduce(
        self,
        *,
        binding: UltraEPLayerBinding,
        reduce_event: object,
    ) -> None:
        self.external_manager.wait_grad_reduce(reduce_event)
        restore_fp32_master_grads_to_fsdp(binding.experts)

    def release_slot(self, binding: UltraEPLayerBinding, slot: object) -> None:
        # The external API allocates virtual IDs cyclically and has no storage lease object;
        # this Adapter owns the active-call guard and permits reuse only after completion.
        del slot
        self._active_layer_calls.remove(binding.layer_fqn)

    def close(self) -> None:
        self.external_manager.destroy()


class _UltraEPInvocation:
    """一个调用唯一拥有 placement、slot、events 和 backward ordering。"""

    def __init__(
        self,
        *,
        manager: UltraEPManagerAdapter,
        binding: UltraEPLayerBinding,
        slot: object,
    ) -> None:
        self.manager = manager
        self.binding = binding
        self.slot = slot
        self.placement: object
        self.weight_event: object
        self.reduce_event: object
        self._released = False

    def prepare_physical_dispatch(
        self,
        *,
        logical_ids: Tensor,
        logical_counts: Tensor,
    ) -> Tensor:
        self.placement = self.manager.update_placement(
            binding=self.binding,
            slot=self.slot,
            logical_ids=logical_ids,
            logical_counts=logical_counts,
        )
        self.manager.refresh_master_pointers(self.binding)
        self.weight_event = self.manager.weight_sync(
            slot=self.slot,
            placement=self.placement,
            async_finish=True,
        )
        return self.manager.reroute(
            slot=self.slot,
            placement=self.placement,
            logical_ids=logical_ids,
        )

    def wait_weights_at_first_consumer(self) -> None:
        # Real code inserts current-stream wait_event; it does not synchronize the host.
        current_stream_wait_event(self.weight_event)

    def expert_execution(self) -> ExpertExecution:
        return self.manager.replica_execution(slot=self.slot)

    def replay_weights_for_backward(self) -> None:
        self.manager.replay_weights(slot=self.slot, placement=self.placement)

    def start_grad_reduce(self) -> None:
        self.reduce_event = self.manager.start_grad_reduce(
            binding=self.binding,
            slot=self.slot,
            placement=self.placement,
        )

    def wait_grad_reduce_restore_and_release(self) -> None:
        self.manager.finish_grad_reduce(
            binding=self.binding,
            reduce_event=self.reduce_event,
        )
        self.release()

    def release(self) -> None:
        if not self._released:
            self.manager.release_slot(self.binding, self.slot)
            self._released = True


class _UltraEPGradReduceJoin(torch.autograd.Function):
    """Forward 位于 attention 前；backward 在 FSDP completion boundary 前 join。"""

    @staticmethod
    def forward(ctx: Any, layer_input: Tensor, invocation: _UltraEPInvocation) -> Tensor:
        ctx.invocation = invocation
        return layer_input

    @staticmethod
    def backward(ctx: Any, grad_input: Tensor) -> tuple[Tensor, None]:
        ctx.invocation.wait_grad_reduce_restore_and_release()
        return grad_input, None


class _UltraEPGradReduceStart(torch.autograd.Function):
    """Forward 位于 dispatch 前；backward 在 expert/dispatch backward 后启动 reduce。"""

    @staticmethod
    def forward(ctx: Any, hidden_states: Tensor, invocation: _UltraEPInvocation) -> Tensor:
        ctx.invocation = invocation
        return hidden_states

    @staticmethod
    def backward(ctx: Any, grad_hidden: Tensor) -> tuple[Tensor, None]:
        ctx.invocation.start_grad_reduce()
        return grad_hidden, None


class _UltraEPWeightSyncForBackward(torch.autograd.Function):
    """Forward 位于 expert output 后；backward 在 expert DGrad 前 replay replicas。"""

    @staticmethod
    def forward(ctx: Any, expert_output: Tensor, invocation: _UltraEPInvocation) -> Tensor:
        ctx.invocation = invocation
        return expert_output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        ctx.invocation.replay_weights_for_backward()
        return grad_output, None


class UltraEPRuntime(EPExecutionRuntime):
    """唯一拥有 Manager；不使用 Provider + global registry 双缓存。"""

    def __init__(self, *, config: Any, ep_group: Any) -> None:
        self.config = config
        self.ep_group = ep_group
        validate_ultraep_python_contract(config, ep_group)
        self.num_experts = config.n_routed_experts
        self.replica_slots_per_rank = config.ultraep_cfg.replica_slots_per_rank
        self.ep_size = ep_group.size()
        self._closed = False
        self.manager: UltraEPManagerAdapter | None = None

    @property
    def num_physical_experts(self) -> int:
        # DeepEP global IDs are rank blocks [B masters, R replicas]; master IDs are
        # interleaved with replica gaps and are not the logical range [0, E).
        return self.num_experts + self.ep_size * self.replica_slots_per_rank

    def bind_dispatcher(self, *, layer_fqn: str, experts: MoEBlock) -> GenericDispatcher:
        # Binding object can be resource-free during meta build; real manager handle is installed later.
        binding = UltraEPLayerBinding(layer_fqn, experts, external_layer_handle=None)
        inner = build_deepep_dispatcher(
            n_routed_experts=self.num_physical_experts,
            ep_group=self.ep_group,
        )
        return UltraEPDispatcher(runtime=self, binding=binding, inner=inner)

    def validate_before_fsdp(self, *, model: nn.Module, fsdp_config: Any) -> None:
        validate_ultraep_fsdp_config(model, fsdp_config)
        validate_dual_gmm_supports_configured_replica_layout(self.config)

    def install_after_fsdp(self, *, model: nn.Module, fsdp_config: Any) -> None:
        self.manager = UltraEPManagerAdapter(config=self.config, ep_group=self.ep_group)
        for dispatcher in iter_ultraep_dispatchers(model):
            dispatcher.binding = self.manager.register_layer(
                dispatcher.binding.layer_fqn,
                dispatcher.binding.experts,
            )

    def begin_invocation(self, binding: UltraEPLayerBinding) -> _UltraEPInvocation:
        manager = cast(UltraEPManagerAdapter, self.manager)
        return _UltraEPInvocation(
            manager=manager,
            binding=binding,
            slot=manager.lease_slot(binding),
        )

    def close(self) -> None:
        if not self._closed:
            if self.manager is not None:
                self.manager.close()
            self._closed = True


class UltraEPDispatcher(GenericDispatcher):
    """UltraEP control plane around an unchanged DeepEP six-stage Implementation。"""

    def __init__(
        self,
        *,
        runtime: UltraEPRuntime,
        binding: UltraEPLayerBinding,
        inner: GenericDispatcher,
    ) -> None:
        self.runtime = runtime
        self.binding = binding
        self.inner = inner

    def begin_invocation(self, layer_input: Tensor) -> RoutedExpertInvocation:
        invocation = self.runtime.begin_invocation(self.binding)
        joined_input = _UltraEPGradReduceJoin.apply(layer_input, invocation)
        return RoutedExpertInvocation(
            dispatcher=self,
            layer_input=joined_input,
            backend_state=invocation,
        )

    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        invocation_state: object | None,
        async_op: bool = False,
    ) -> StageResult:
        invocation = cast(_UltraEPInvocation, invocation_state)
        physical_ids = invocation.prepare_physical_dispatch(
            logical_ids=topk_ids,
            logical_counts=tokens_per_expert,
        )
        dispatch_input = _UltraEPGradReduceStart.apply(hidden_states, invocation)
        inner_pre = self.inner.dispatch_preprocess(
            hidden_states=dispatch_input,
            topk_ids=physical_ids,
            topk_weights=topk_weights,
            # The inner DeepEP Adapter currently derives physical counts from IDs.
            # It accepts this required source-count argument but does not reinterpret it.
            tokens_per_expert=tokens_per_expert,
            invocation_state=None,
            async_op=async_op,
        )
        return {"inner": inner_pre, "_invocation": invocation}

    def dispatch(
        self,
        *,
        pre_dispatched: StageResult,
        topk_weights: Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        return {
            "inner": self.inner.dispatch(
                pre_dispatched=pre_dispatched["inner"],
                topk_weights=topk_weights,
                async_op=async_op,
                decoding=decoding,
            )
        }

    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        async_op: bool = False,
    ) -> StageResult:
        invocation = cast(_UltraEPInvocation, pre_dispatched["_invocation"])
        inner_post = self.inner.dispatch_postprocess(
            pre_dispatched=pre_dispatched["inner"],
            dispatched=dispatched["inner"],
            async_op=async_op,
        )
        invocation.wait_weights_at_first_consumer()
        return {
            "hidden_states": inner_post["hidden_states"],
            # DeepEP local order is fixed to [B master groups, R replica groups].
            "tokens_per_expert": inner_post["tokens_per_expert"],
            "expert_execution": invocation.expert_execution(),
            "inner": inner_post,
        }

    def combine_preprocess(
        self,
        *,
        hidden_states: Tensor,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        invocation = cast(_UltraEPInvocation, pre_dispatched["_invocation"])
        replay_edge = _UltraEPWeightSyncForBackward.apply(hidden_states, invocation)
        inner_pre_combined = self.inner.combine_preprocess(
            hidden_states=replay_edge,
            pre_dispatched=pre_dispatched["inner"],
            dispatched=dispatched["inner"],
            post_dispatched=post_dispatched["inner"],
            async_op=async_op,
            decoding=decoding,
        )
        return {"inner": inner_pre_combined}

    def combine(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        pre_combined: StageResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> StageResult:
        return {
            "inner": self.inner.combine(
                pre_dispatched=pre_dispatched["inner"],
                dispatched=dispatched["inner"],
                post_dispatched=post_dispatched["inner"],
                pre_combined=pre_combined["inner"],
                async_op=async_op,
                decoding=decoding,
            )
        }

    def combine_postprocess(
        self,
        *,
        pre_dispatched: StageResult,
        dispatched: StageResult,
        post_dispatched: StageResult,
        pre_combined: StageResult,
        combined: StageResult,
        async_op: bool = False,
    ) -> StageResult:
        invocation = cast(_UltraEPInvocation, pre_dispatched["_invocation"])
        inner_post = self.inner.combine_postprocess(
            pre_dispatched=pre_dispatched["inner"],
            dispatched=dispatched["inner"],
            post_dispatched=post_dispatched["inner"],
            pre_combined=pre_combined["inner"],
            combined=combined["inner"],
            async_op=async_op,
        )
        if not torch.is_grad_enabled():
            invocation.release()
        return {"hidden_states": inner_post["hidden_states"]}


# =============================================================================
# 6. [CHANGED EXPERT COMPUTE] 普通路径不接收动态 backend kwargs
# =============================================================================


class GroupedLinear(nn.Module):
    """仅展示 public old path 与 dynamic BF16 capability。"""

    weight: nn.Parameter
    local_out_features: int
    local_in_features: int

    def forward(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        decoding: bool = False,
    ) -> Tensor:
        # Existing public Interface used by ordinary BF16/FP8/CUTLASS/NPU substitutes.
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        weight = weight.view(-1, self.local_out_features, self.local_in_features)
        return existing_group_gemm(
            hidden_states,
            weight,
            tokens_per_expert,
            decoding=decoding,
        )

    def forward_with_execution(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        execution: ProjectionExecution,
    ) -> Tensor:
        """Only validated dynamic BF16 configs call this concrete capability。"""
        if execution.primary_weight is None:
            primary = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
            primary = primary.view(-1, self.local_out_features, self.local_in_features)
        else:
            # MoonEP already supplies a shaped contiguous local [2B, out, in] VMM view.
            primary = execution.primary_weight

        if execution.secondary_weight is None:
            if execution.primary_grad_out is None:
                return existing_group_gemm(hidden_states, primary, tokens_per_expert)
            return group_gemm_with_direct_grad_out(
                hidden_states,
                primary,
                tokens_per_expert,
                grad_weight_out=execution.primary_grad_out,
            )

        # Kernel uses the external storage contract directly. It must not call contiguous().
        return dual_allocation_group_gemm(
            hidden_states,
            primary_weight=primary,
            secondary_weight=execution.secondary_weight,
            tokens_per_expert=tokens_per_expert,
            primary_grad_out=execution.primary_grad_out,
            secondary_grad_out=execution.secondary_grad_out,
            secondary_expert_stride=execution.secondary_weight.stride(0),
        )


class MoEBlock(nn.Module):
    """不出现 MoonEP/UltraEP backend name。"""

    def __init__(self, fused_w1w3: nn.Module, fused_w2: nn.Module, moe_act: nn.Module) -> None:
        super().__init__()
        self.fused_w1w3 = fused_w1w3
        self.fused_w2 = fused_w2
        self.moe_act = moe_act

    def forward(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        *,
        decoding: bool,
        execution: ExpertExecution | None,
    ) -> Tensor:
        if execution is None:
            # Crucial: no new kwargs reach TileWise FP8/CUTLASS/NPU substitutes.
            gate_up = self.fused_w1w3(hidden_states, tokens_per_expert, decoding)
            activated = self.moe_act(gate_up)
            return self.fused_w2(activated, tokens_per_expert, decoding)

        # Config validation guarantees dynamic paths use this concrete capability.
        w1w3 = cast(GroupedLinear, self.fused_w1w3)
        w2 = cast(GroupedLinear, self.fused_w2)
        gate_up = w1w3.forward_with_execution(
            hidden_states,
            tokens_per_expert,
            execution.gate_up,
        )
        return w2.forward_with_execution(
            self.moe_act(gate_up),
            tokens_per_expert,
            execution.down,
        )


# =============================================================================
# 7. [CLIENT] Decoder 单 microbatch 与 Domino 的主要流程
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
        # Only lifecycle extension outside the six post-router stages.
        invocation = self.dispatcher.begin_invocation(layer_input)

        residual, routed_hidden, router = self._pre_moe_forward(
            hidden_states=invocation.layer_input,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
        )
        origin_shape = routed_hidden.shape

        invocation.dispatch_preprocess(
            hidden_states=routed_hidden.view(-1, routed_hidden.shape[-1]),
            topk_ids=router["topk_ids"],
            topk_weights=router["topk_weights"],
            tokens_per_expert=router["tokens_per_expert"],
        )
        invocation.dispatch()
        batch = invocation.dispatch_postprocess()

        expert_output = self.experts(
            batch.hidden_states,
            batch.tokens_per_expert,
            decoding=False,
            execution=batch.execution,
        )
        invocation.combine_preprocess(expert_output, async_op=True)
        invocation.combine(async_op=True)

        # Existing overlap stays: routed combine runs while shared experts compute.
        shared_output = self._shared_experts_forward(routed_hidden) if self.n_shared_experts > 0 else None
        routed_output = invocation.combine_postprocess(async_op=True).view(*origin_shape)
        output = self._post_moe_forward(
            combined_hidden_states=routed_output,
            residual=residual,
            shared_experts_out=shared_output,
        )
        # Public return remains logical; UltraEP physical IDs never leave its invocation.
        return output, router["logits"], router["router_weights"], router["topk_ids"]

    def _micro_batch_forward(
        self,
        layer_inputs: Sequence[Tensor],
        seq_ctxs: Sequence[Any],
        position_embeddings: Sequence[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, ...]:
        # This common scheduler is enabled only for backends whose validation declares
        # intra-layer concurrency. MoonEP uses its static ordinal ring; UltraEP v1 rejects it.
        invocations: list[RoutedExpertInvocation] = []
        residuals: list[Tensor] = []
        routed_hiddens: list[Tensor] = []
        routers: list[RouterResults] = []

        # Attention + logical router + phase 1. Each call leases independent state.
        for layer_input, seq_ctx, pos_emb in zip(layer_inputs, seq_ctxs, position_embeddings):
            invocation = self.dispatcher.begin_invocation(layer_input)
            residual, routed_hidden, router = self._pre_moe_forward(
                hidden_states=invocation.layer_input,
                seq_ctx=seq_ctx,
                position_embeddings=pos_emb,
            )
            invocation.dispatch_preprocess(
                hidden_states=routed_hidden.view(-1, routed_hidden.shape[-1]),
                topk_ids=router["topk_ids"],
                topk_weights=router["topk_weights"],
                tokens_per_expert=router["tokens_per_expert"],
                async_op=True,
            )
            invocations.append(invocation)
            residuals.append(residual)
            routed_hiddens.append(routed_hidden)
            routers.append(router)

        # Phases 2-4 and expert compute.
        for invocation in invocations:
            invocation.dispatch(async_op=True)
            batch = invocation.dispatch_postprocess(async_op=True)
            expert_output = self.experts(
                batch.hidden_states,
                batch.tokens_per_expert,
                decoding=False,
                execution=batch.execution,
            )
            invocation.combine_preprocess(expert_output, async_op=True)

        # Phase 5 is launched for all microbatches before shared expert compute.
        for invocation in invocations:
            invocation.combine(async_op=True)

        shared_outputs = [
            self._shared_experts_forward(hidden) if self.n_shared_experts > 0 else None for hidden in routed_hiddens
        ]

        outputs: list[Tensor] = []
        for invocation, residual, routed_hidden, shared_output in zip(
            invocations,
            residuals,
            routed_hiddens,
            shared_outputs,
        ):
            routed_output = invocation.combine_postprocess(async_op=True).view_as(routed_hidden)
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
# 8. [CLIENT] Config、model build、FSDP install 与 teardown
# =============================================================================


class MoEConfig:
    dispatcher: DispatcherName
    n_routed_experts: int
    training_dtype: Literal["bf16", "fp8"]
    expert_tp_size: int
    data_parallel_size: int
    intra_layer_micro_batch: int
    mtp_config: Any | None
    moe_bias: bool
    ultraep_cfg: Any | None
    moonep_cfg: Any | None


def build_ep_runtime(
    *,
    config: MoEConfig,
    ep_group: Any,
) -> EPExecutionRuntime | None:
    """One discriminator; UltraEP is not deepep + an orthogonal decoder overlay。"""
    if config.dispatcher == "moonep":
        return MoonEPRuntime(config=config, ep_group=ep_group)
    if config.dispatcher == "ultraep":
        return UltraEPRuntime(config=config, ep_group=ep_group)
    return None


def build_layer_dispatcher(
    *,
    config: MoEConfig,
    ep_group: Any,
    runtime: EPExecutionRuntime | None,
    layer_fqn: str,
    experts: MoEBlock,
) -> GenericDispatcher:
    if runtime is not None:
        return runtime.bind_dispatcher(layer_fqn=layer_fqn, experts=experts)
    return build_existing_dispatcher(
        dispatcher=config.dispatcher,
        n_routed_experts=config.n_routed_experts,
        ep_group=ep_group,
    )


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
            dispatcher_builder=lambda layer_fqn, experts: build_layer_dispatcher(
                config=config,
                ep_group=ep_group,
                runtime=self._ep_runtime,
                layer_fqn=layer_fqn,
                experts=experts,
            ),
        )

    def fully_shard(self, fsdp_config: Any) -> MoEModel:
        # Critical ordering: reject unsupported combinations before mutating parameters.
        if self._ep_runtime is not None:
            self._ep_runtime.validate_before_fsdp(model=self, fsdp_config=fsdp_config)

        existing_fully_shard_model(self, fsdp_config)

        if self._ep_runtime is not None:
            self._ep_runtime.install_after_fsdp(model=self, fsdp_config=fsdp_config)
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
# 9. [KERNEL CONTRACT] dual allocation 不复制 official strided replicas
# =============================================================================


def dual_allocation_group_gemm(
    hidden_states: Tensor,
    *,
    primary_weight: Tensor,
    secondary_weight: Tensor,
    tokens_per_expert: Tensor,
    primary_grad_out: Tensor | None,
    secondary_grad_out: Tensor | None,
    secondary_expert_stride: int,
) -> Tensor:
    """伪代码 contract；真实 op 实现 forward、DGrad 和两类 WGrad。

    Group ordering:
        tokens_per_expert[:primary_weight.shape[0]] -> primary allocation
        remaining groups -> secondary allocation

    ``secondary_expert_stride`` comes from the real view. The op must not assert that
    ``secondary_weight.is_contiguous()`` and must not create a contiguous snapshot.
    """
    return launch_stride_aware_dual_gmm(
        hidden_states,
        primary_weight,
        secondary_weight,
        tokens_per_expert,
        primary_grad_out,
        secondary_grad_out,
        secondary_expert_stride,
    )


# =============================================================================
# 10. Omitted production implementations / external boundaries
# =============================================================================


def validate_moonep_python_contract(config: Any, ep_group: Any) -> None: ...


def register_moonep_layer(runtime: MoonEPRuntime, layer_fqn: str, experts: MoEBlock) -> object: ...


def create_moonep_invocation(
    runtime: MoonEPRuntime,
    layer_binding: object,
) -> _MoonEPInvocation: ...


def materialize_moonep_local_tensors(
    invocation: _MoonEPInvocation,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: ...


def validate_moonep_fsdp_config(model: nn.Module, fsdp_config: Any) -> None: ...


def install_moonep_vmm_and_fsdp_landing(
    runtime: MoonEPRuntime,
    model: nn.Module,
    fsdp_config: Any,
) -> None: ...


def destroy_moonep_buffers_vmm_and_landing(runtime: MoonEPRuntime) -> None: ...


def create_external_ultraep_manager(
    *,
    config: Any,
    ep_group: Any,
    explicitly_destroy: bool,
) -> Any: ...


def refresh_external_master_pointers(manager: Any, experts: MoEBlock) -> None: ...


def stage_master_grads_to_fp32(experts: MoEBlock) -> None: ...


def restore_fp32_master_grads_to_fsdp(experts: MoEBlock) -> None: ...


def current_stream_wait_event(event: object) -> None: ...


def validate_ultraep_python_contract(config: Any, ep_group: Any) -> None:
    unsupported = (
        ep_group.size() <= 1
        or config.ultraep_cfg is None
        or config.training_dtype != "bf16"
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


def validate_ultraep_fsdp_config(model: nn.Module, fsdp_config: Any) -> None:
    """UltraEP v1 keeps the PR's deliberately narrow, proven capability boundary。"""
    del model
    if fsdp_config.recompute_ratio > 0:
        raise ValueError("UltraEP v1 does not support activation recompute")


def validate_dual_gmm_supports_configured_replica_layout(config: Any) -> None: ...


def iter_ultraep_dispatchers(model: nn.Module) -> Sequence[UltraEPDispatcher]: ...


def build_deepep_dispatcher(*, n_routed_experts: int, ep_group: Any) -> GenericDispatcher: ...


def existing_group_gemm(
    hidden_states: Tensor,
    weight: Tensor,
    tokens_per_expert: Tensor,
    *,
    decoding: bool = False,
) -> Tensor: ...


def group_gemm_with_direct_grad_out(
    hidden_states: Tensor,
    weight: Tensor,
    tokens_per_expert: Tensor,
    *,
    grad_weight_out: Tensor,
) -> Tensor: ...


def launch_stride_aware_dual_gmm(*args: Any) -> Tensor: ...


def build_existing_dispatcher(
    *,
    dispatcher: DispatcherName,
    n_routed_experts: int,
    ep_group: Any,
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
# - UltraEP v1 rejects a second same-layer invocation before shared storage is overwritten;
# - each backend only enables Domino/MTP/reentrant paths declared by its capability boundary;
# - ordinary FP8/CUTLASS/NPU paths still call their original three-argument Interface;
# - DCP/HF state excludes all runtime/invocation tensors and cold-runtime restore works;
# - explicit close permits rebuilding a different runtime in the same process.
