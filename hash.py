import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rff(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    z = x @ weight + bias
    cos_part = torch.cos(z)
    sin_part = torch.sin(z)
    feats = torch.cat([cos_part, sin_part], dim=-1)
    scale = 1.0 / math.sqrt(weight.size(-1))
    return feats * scale


class CausalSegmentationSetHash(nn.Module):
    """Causal, order-insensitive hashing with multiscale scrambling.

    The module computes a deepset-style fingerprint of a prefix ``x[:t]`` for every
    timestep ``t``. It uses random Fourier features for robustness, exponential
    segment priors for speed, and a controllable softness parameter that lets the
    hash gradually retreat toward an anchor when evidence is weak.
    """

    def __init__(
        self,
        c: int,
        d_rff: int = 256,
        n_scales: int = 4,
        base_length: float = 4.0,
        softness: float = 0.35,
        anchor_strength: float = 2.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if d_rff % 2 != 0:
            raise ValueError("d_rff must be even to pair cosine and sine features.")

        self.c = c
        self.d_rff = d_rff
        self.n_scales = n_scales
        self.base_length = float(base_length)
        self.softness = float(softness)
        self.anchor_strength = float(anchor_strength)

        generator = torch.Generator()
        generator.manual_seed(seed)

        half = d_rff // 2
        self.register_buffer("weight", torch.randn(c, half, generator=generator) / math.sqrt(c))
        self.register_buffer("bias", 2 * math.pi * torch.rand(half, generator=generator))

        self.phi = nn.Sequential(
            nn.Linear(d_rff, c, bias=True),
            nn.SiLU(),
            nn.Linear(c, c, bias=True),
        )

        self.anchor = nn.Parameter(torch.zeros(n_scales, c))

        length_scales = [self.base_length * (2.0 ** s) for s in range(n_scales)]
        self.register_buffer("length_scales", torch.tensor(length_scales))

        scramble = torch.empty(n_scales, d_rff, generator=generator)
        scramble.uniform_(-1.0, 1.0)
        scramble = torch.sign(scramble)
        self.register_buffer("scramble_sign", scramble)

        permutations = []
        for s in range(n_scales):
            perm = torch.randperm(d_rff, generator=generator)
            permutations.append(perm)
        self.register_buffer("scramble_perm", torch.stack(permutations, dim=0))

        self.mix = nn.Linear(n_scales * c, c, bias=True)

    def forward(self, x: torch.Tensor, softness: Optional[float] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("expected input shaped (B, T, C)")
        batch, time, channels = x.shape
        if channels != self.c:
            raise ValueError("channel dimension does not match constructor value")

        device = x.device
        dtype = x.dtype

        working_softness = self.softness if softness is None else float(softness)
        if working_softness <= 0.0:
            raise ValueError("softness must be positive")

        u = _rff(x.reshape(batch * time, channels), self.weight, self.bias)
        u = u.view(batch, time, self.d_rff)

        reps = []
        for s in range(self.n_scales):
            decay = math.exp(-1.0 / float(self.length_scales[s]))
            scramble_sign = self.scramble_sign[s].to(dtype=dtype, device=device)
            scramble_perm = self.scramble_perm[s].to(device=device)

            scrambled = u * scramble_sign
            scrambled = torch.gather(scrambled, -1, scramble_perm.view(1, 1, -1).expand(batch, time, -1))

            running_sum = torch.zeros(batch, self.d_rff, device=device, dtype=dtype)
            running_count = torch.zeros(batch, 1, device=device, dtype=dtype)
            steps = []
            counts = []

            for t in range(time):
                running_sum = running_sum * decay + scrambled[:, t]
                running_count = running_count * decay + 1.0
                mean = running_sum / running_count.clamp(min=1e-6)
                steps.append(mean)
                counts.append(running_count)

            stacked = torch.stack(steps, dim=1)
            stacked_counts = torch.stack(counts, dim=1)
            segment_embed = self.phi(stacked)

            confidence = stacked_counts / (self.length_scales[s] + 1e-6)
            confidence = confidence / working_softness
            confidence = torch.tanh(confidence)

            anchor = self.anchor[s].to(device=device, dtype=dtype).view(1, 1, self.c)
            hardened = anchor + (segment_embed - anchor) * confidence
            hardened = hardened / (1.0 + self.anchor_strength)
            hardened = hardened + anchor * (self.anchor_strength / (1.0 + self.anchor_strength))

            reps.append(hardened)

        concatenated = torch.cat(reps, dim=-1)
        mixed = self.mix(concatenated)
        return mixed
