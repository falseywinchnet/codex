import torch

from hash import CausalSegmentationSetHash


def test_hash_smoke():
    torch.manual_seed(0)
    module = CausalSegmentationSetHash(c=8, d_rff=32, n_scales=3, base_length=3.0)
    x = torch.randn(2, 5, 8)
    output = module(x)

    assert output.shape == (2, 5, 8)
    assert torch.isfinite(output).all()


def test_softness_retreats_toward_anchor():
    torch.manual_seed(1)
    module = CausalSegmentationSetHash(c=4, d_rff=32, n_scales=2, anchor_strength=1.5)

    seq_a = torch.randn(1, 6, 4)
    seq_b = torch.flip(seq_a, dims=[1])

    output_hard_a = module(seq_a, softness=0.2)
    output_hard_b = module(seq_b, softness=0.2)

    output_soft_a = module(seq_a, softness=1.2)
    output_soft_b = module(seq_b, softness=1.2)

    hard_gap = torch.norm(output_hard_a - output_hard_b, dim=-1).mean()
    soft_gap = torch.norm(output_soft_a - output_soft_b, dim=-1).mean()

    anchor = module.anchor.mean(dim=0)
    hard_anchor_distance = torch.norm(output_hard_a - anchor, dim=-1).mean()
    soft_anchor_distance = torch.norm(output_soft_a - anchor, dim=-1).mean()

    assert soft_gap < hard_gap
    assert soft_anchor_distance < hard_anchor_distance
