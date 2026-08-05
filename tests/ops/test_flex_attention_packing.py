import os
import subprocess
import sys
import textwrap


def test_many_document_packing_mask_compiles_without_expanding_each_document() -> None:
    script = textwrap.dedent(
        """
        import torch

        from xtuner.v1.ops.attn_imp import create_packing_block_causal_mask

        cu_seqlens = torch.arange(257, dtype=torch.int32) * 256

        def build_mask(cu_seqlens):
            return create_packing_block_causal_mask(cu_seqlens, sequence_length=65536).kv_num_blocks

        compiled = torch.compile(build_mask, backend="eager", fullgraph=True)
        assert compiled(cu_seqlens).numel() > 0

        mask = create_packing_block_causal_mask(cu_seqlens, sequence_length=65536)
        assert bool(mask.mask_mod(0, 0, 255, 0))
        assert not bool(mask.mask_mod(0, 0, 256, 255))
        assert not bool(mask.mask_mod(0, 0, 0, 1))
        """
    )

    # A subprocess gives graph capture a hard upper bound. The old Python loop
    # expanded 256 torch.full calls and spent minutes in Dynamo/SymPy here.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30, env=env)
