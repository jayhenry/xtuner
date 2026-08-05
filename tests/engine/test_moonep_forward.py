import unittest

import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model.moe.glm52 import Glm52MoEConfig
from xtuner.v1.model.moe.moe import MoEConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import DSAMLAConfig, MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig, NoAuxRouterConfig


def _tiny_config(family: str, dispatcher: str, *, compile: bool) -> MoEConfig:
    common = dict(
        vocab_size=256,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        # With EP4/E8 each home chunk must satisfy CUDA's 2 MiB VMM
        # granularity for both fused projections.
        hidden_size=512,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=1024,
        ep_size=4,
        dispatcher=dispatcher,
        moonep_staging_reference=dispatcher == "moonep",
        balancing_loss_cfg=None,
        compile_cfg=(
            {"xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": {"fullgraph": True}}
            if compile
            else False
        ),
    )
    if family == "qwen":
        return Qwen3MoEConfig(
            **common,
            bos_token_id=2,
            attention=MHAConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                qk_norm=True,
                attn_impl="flex_attention",
            ),
            router=GreedyRouterConfig(
                scoring_func="softmax",
                norm_topk_prob=True,
                router_scaling_factor=1.0,
            ),
        )
    if family == "glm52":
        return Glm52MoEConfig(
            **common,
            hf_eos_token_id=[1],
            attention=DSAMLAConfig(
                num_attention_heads=2,
                head_dim=4,
                kv_lora_rank=4,
                q_lora_rank=8,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
                index_topk=4,
                index_head_dim=4,
                index_n_heads=2,
                indexer_types=["full", "shared", "shared"],
                sparse_mla_backend="torch",
            ),
            hf_head_dim=4,
            qk_head_dim=8,
            router=NoAuxRouterConfig(
                n_group=1,
                topk_group=1,
                scoring_func="sigmoid",
                norm_topk_prob=True,
                router_scaling_factor=2.5,
            ),
            mlp_layer_types=["dense", "sparse", "sparse"],
            num_nextn_predict_layers=None,
        )
    raise AssertionError(f"unknown tiny model family: {family}")


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires 8 CUDA devices")
class TestMoonEPStagingForward(DeterministicDDPTestCase):
    def _forward(self, family: str, dispatcher: str) -> torch.Tensor:
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config(family, dispatcher, compile=True),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0, torch_compile=True),
            intra_layer_micro_batch=1,
        )
        engine.init_model_weights()
        if dispatcher == "moonep":
            assert engine.model.config.intra_layer_micro_batch == 1
        input_ids = torch.arange(2, 18, device="cuda").view(1, -1)

        try:
            engine.model.eval()
            with torch.no_grad():
                first = engine.model(
                    seq_ctx=SequenceContext.from_input_ids((input_ids,), device="cuda"),
                    loss_ctx=None,
                ).logits

            assert first is not None
            assert torch.isfinite(first).all()
            repeats = 3 if dispatcher == "moonep" else int(dispatcher != "deepep")
            for _ in range(repeats):
                with torch.no_grad():
                    repeated = engine.model(
                        seq_ctx=SequenceContext.from_input_ids((input_ids,), device="cuda"),
                        loss_ctx=None,
                    ).logits
                torch.testing.assert_close(first, repeated, rtol=0, atol=0)
            return first.clone()
        finally:
            # Resource teardown may unmap VMM landings, so the test must first
            # complete queued output copies. This is lifecycle-only, not a hot-path sync.
            torch.cuda.synchronize()
            if dispatcher == "moonep":
                engine.model.destroy_moonep()
            del engine
            # DeepEP owns a process-scoped C++ Buffer. Forcing cyclic GC here
            # can destruct it on only a subset of ranks; leave that resource
            # to the distributed process teardown.
            torch.cuda.empty_cache()
            dist.barrier()

    def _assert_matches_reference(self, family: str, reference: str) -> None:
        self.create_pg("cuda")
        expected = self._forward(family, reference)
        moonep = self._forward(family, "moonep")
        torch.testing.assert_close(moonep, expected, rtol=1e-2, atol=1e-2)

    def test_qwen_fixed_length_fused_expert_forward_matches_deepep(self) -> None:
        self._assert_matches_reference("qwen", "deepep")

    def test_glm52_fixed_length_fused_expert_forward_matches_all2all(self) -> None:
        self._assert_matches_reference("glm52", "all2all")

    @property
    def world_size(self) -> int:
        return 8
