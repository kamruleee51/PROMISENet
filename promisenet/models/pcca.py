"""
Prompt-Conditioned Channel Attention (PCCA) and the bounding-box prompt encoder.

Faithful extraction from the notebooks confirmed (by exact match against the paper's reported
Table results, see the audit in the source repo's agent_files/) to be the canonical source for
PROMISE-CNN and PROMISE-Txformer:
    Experiments_2.0/{dataset}/MultiGateAttention/.../SE/*.ipynb   (PROMISE-CNN)
    Unetr_monai/gated_unetr/{dataset}/*.ipynb                     (PROMISE-Txformer)
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class PromptSEFusion_gate(nn.Module):
    """PCCA: squeeze-and-excitation-style gating conditioned on both image and prompt features.

    Implements the paper's Eq. 8 "Feature Modulation Unit" in full:
        F_out = F_b * sigmoid(MLP(proj(F_b) + proj(F_p)))  +  F_b * F_p
    i.e. the channel-wise gate term plus the raw prompt-multiplication term, both present.
    """

    def __init__(self, bottleneck_dim: int, prompt_dim: int, reduction: int = 16):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        self.prompt_proj = nn.Linear(prompt_dim, bottleneck_dim)
        self.bottleneck_proj = nn.Linear(bottleneck_dim, bottleneck_dim)

        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim // reduction, bottleneck_dim),
            nn.Sigmoid(),
        )

    def forward(self, bottleneck: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = bottleneck.shape

        bottleneck_vec = bottleneck.mean(dim=[2, 3])  # (B, C)
        prompt_vec = prompt.mean(dim=[2, 3])          # (B, C)

        bottleneck_vec = self.bottleneck_proj(bottleneck_vec)
        prompt_vec = self.prompt_proj(prompt_vec)

        fused = bottleneck_vec + prompt_vec
        weights = self.fusion(fused).view(B, C, 1, 1)

        return bottleneck * weights + bottleneck * prompt


class PositionEmbeddingRandom(nn.Module):
    """Fourier positional encoding, identical to SAM's implementation."""

    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0):
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        return self._pe_encoding(
            torch.stack([x_embed / w, y_embed / h], dim=-1)
        ).permute(2, 0, 1)


class BoxPositionEncoder(nn.Module):
    """Bounding-box prompt encoder: box(es) -> dense (embed_dim, H, W) feature map.

    Each stage of PROMISE-CNN / PROMISE-Txformer instantiates its OWN independently-weighted
    BoxPositionEncoder, hardcoded to that stage's exact (embed_dim, feature_size) — this is a
    genuine deviation from the paper's described "encode once, resize per level" Prompt Encoder
    (Eqs. eq:fbox-eq:promptscale), which was verified in the source repo and is preserved here
    rather than "corrected" to the paper's literal description, since it's what actually produced
    the reported numbers.

    Accepts boxes as [B, N, 4] to support both single-box datasets (ISIC, Kvasir; N=1) and
    multi-box datasets (CAMUS: one box per cardiac chamber).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_size: Tuple[int, int] = (256, 256),
        feature_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.feature_size = feature_size

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.box_embed = nn.Embedding(1, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: [B, N, 4] tensor of (x1, y1, x2, y2) coordinates, in the original image's
                   pixel space. A box of all zeros is treated as "no box" and skipped (padding).
        Returns:
            dense_encoding: [B, embed_dim, H, W]
        """
        B, N, _ = boxes.shape
        device = boxes.device

        dense_encoding = torch.zeros(B, self.embed_dim, *self.feature_size, device=device)

        scale_h = self.feature_size[0] / self.image_size[0]
        scale_w = self.feature_size[1] / self.image_size[1]
        boxes = boxes * torch.tensor([scale_w, scale_h, scale_w, scale_h], device=device)

        for b in range(B):
            for n in range(N):
                x1, y1, x2, y2 = boxes[b, n].round().int()

                if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0:
                    continue  # padding box, no real prompt here

                mask = torch.zeros(1, *self.feature_size, device=device)
                mask[:, y1:y2, x1:x2] = 1

                pe = self.pe_layer(self.feature_size)  # [C, H, W]
                box_pe = pe * mask + self.box_embed.weight.view(-1, 1, 1)
                dense_encoding[b] += self.conv(box_pe)

        return dense_encoding
