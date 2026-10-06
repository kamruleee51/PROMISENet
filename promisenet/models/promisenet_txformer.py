"""
PROMISE-Txformer: a custom 12-layer transformer encoder (hand-rolled patch embedding +
nn.TransformerEncoderLayer — NOT MONAI's ViT/UNETR) with a CNN decoder. PCCA is applied at the
bottleneck and every decoder stage. PCCA is deliberately NOT applied at any encoder stage: the
encoder is the transformer itself, and there is no spatial [C,H,W] feature map to gate until the
transformer's final hidden state is reshaped at the bottleneck.

*** CORRECTION NOTICE ***
An earlier version of this file implemented a different architecture (MONAI's ViT + UnetrPrUpBlock/
UnetrUpBlock, gated at 3 encoder-stage projections + the bottleneck, nothing in the decoder) — the
inverse gating pattern from this one. That version was built because its evaluated predictions
matched the paper's reported numbers almost exactly across all four datasets (16/16 metric
comparisons to the reported decimal), which seemed like strong evidence at the time. The author then
directly confirmed that this file (bottleneck + decoder, no encoder gating) is what the paper's
figures and numbers were actually generated from. That earlier metrics match is not yet explained —
most likely the `Metrics_gated_unetr_monai.xlsx` file used for that comparison is stale (from an
earlier run that is no longer the final one), but this has not been independently re-confirmed with a
matching `Metrics_*.xlsx` for *this* architecture. See
agent_files/09_txformer_decoder_discrepancy.md in the source repo for the full history. If you
regenerate evaluation outputs and they don't match the paper, this unexplained gap is the first place
to look.

Faithful extraction from Experiments_2.0/{dataset}/MultiGateAttention/.../SE/*.ipynb (the `prompt_UNetr`
class). Verified identical config across all four datasets: image_size=256, patch_size=16,
hidden_dim=1024, num_layers=12, num_heads=16, mlp_dim=3072, dropout_rate=0.1 — only num_classes and
num_channels vary.
"""
import torch
import torch.nn as nn

from promisenet.models.pcca import BoxPositionEncoder, PromptSEFusion_gate


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class PROMISENet_txformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_channels: int = 3,
        image_size: int = 256,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_dim: int = 3072,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        grid_size = image_size // patch_size  # 256/16 = 16
        num_patches = grid_size * grid_size   # 256

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # The source notebook reshapes transformer-token sequences back into a spatial grid using
        # `patch_size` as the grid's (height, width) — this is only correct because, for this
        # config, grid_size (image_size/patch_size = 16) happens to equal patch_size (16) itself.
        # Preserved here as an explicit `grid_size` (computed correctly, not reusing patch_size by
        # coincidence) so this remains correct if patch_size/image_size are ever changed
        # independently.
        self.grid_size = grid_size

        def image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
            batch_size, num_ch, _, _ = image.shape
            patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            return patches.view(batch_size, -1, patch_size * patch_size * num_ch)

        self.image_to_patches = image_to_patches

        self.patch_embed = nn.Linear(patch_size * patch_size * num_channels, hidden_dim)
        self.register_buffer("positions", torch.arange(start=0, end=num_patches, step=1, dtype=torch.int32))
        self.pos_embed = nn.Embedding(num_patches, hidden_dim)

        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim,
                dropout=dropout_rate, activation=nn.GELU(), batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # CNN decoder: 4 stages. Each stage upsamples the main path (d_i) while a parallel branch
        # (s_i) brings one of the transformer's intermediate hidden states (captured after layers
        # 3/6/9/12, all at the bottleneck's 16x16 resolution) up to the matching resolution via
        # repeated deconv+conv. s4 is the one exception: it processes the raw input image directly
        # (already at full resolution), not a transformer hidden state.
        self.d1 = DeconvBlock(hidden_dim, 512)
        self.s1 = nn.Sequential(DeconvBlock(hidden_dim, 512), ConvBlock(512, 512))
        self.c1 = nn.Sequential(ConvBlock(512 + 512, 512), ConvBlock(512, 512))

        self.d2 = DeconvBlock(512, 384)
        self.s2 = nn.Sequential(
            DeconvBlock(hidden_dim, 384), ConvBlock(384, 384),
            DeconvBlock(384, 384), ConvBlock(384, 384),
        )
        self.c2 = nn.Sequential(ConvBlock(384 + 384, 384), ConvBlock(384, 384))

        self.d3 = DeconvBlock(384, 256)
        self.s3 = nn.Sequential(
            DeconvBlock(hidden_dim, 256), ConvBlock(256, 256),
            DeconvBlock(256, 256), ConvBlock(256, 256),
            DeconvBlock(256, 256), ConvBlock(256, 256),
        )
        self.c3 = nn.Sequential(ConvBlock(256 + 256, 256), ConvBlock(256, 256))

        self.d4 = DeconvBlock(256, 64)
        self.s4 = nn.Sequential(ConvBlock(num_channels, 64), ConvBlock(64, 64))
        self.c4 = nn.Sequential(ConvBlock(64 + 64, 64), ConvBlock(64, 64))

        # PCCA at the bottleneck and every decoder-stage concatenation. Nothing in the encoder.
        self.prompt_encoders = nn.ModuleDict({
            "bottleneck": BoxPositionEncoder(hidden_dim, (image_size, image_size), (grid_size, grid_size)),
            "1024": BoxPositionEncoder(hidden_dim, (image_size, image_size), (grid_size * 2, grid_size * 2)),
            "768": BoxPositionEncoder(768, (image_size, image_size), (grid_size * 4, grid_size * 4)),
            "512": BoxPositionEncoder(512, (image_size, image_size), (grid_size * 8, grid_size * 8)),
            "128": BoxPositionEncoder(128, (image_size, image_size), (grid_size * 16, grid_size * 16)),
        })
        self.fusion_layers = nn.ModuleDict({
            "bottleneck": PromptSEFusion_gate(hidden_dim, hidden_dim, reduction=16),
            "1024": PromptSEFusion_gate(hidden_dim, hidden_dim, reduction=16),
            "768": PromptSEFusion_gate(768, 768, reduction=16),
            "512": PromptSEFusion_gate(512, 512, reduction=16),
            "128": PromptSEFusion_gate(128, 128, reduction=16),
        })

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] input image.
            prompt: [B, N, 4] bounding box(es) in image pixel coordinates.
        Returns:
            [B, num_classes, H, W] softmax probabilities — softmax is applied internally here, the
            same as PROMISE-CNN (unlike the earlier, now-superseded MONAI-based version of this
            file, which returned raw logits).
        """
        inputs = self.image_to_patches(images, self.patch_size)
        x = self.patch_embed(inputs) + self.pos_embed(self.positions)

        skip_connection_index = {3, 6, 9, 12}
        skips = []
        for i in range(self.num_layers):
            x = self.trans_encoder_layers[i](x)
            if (i + 1) in skip_connection_index:
                skips.append(x)
        z3, z6, z9, z12 = skips

        batch = inputs.shape[0]
        shape = (batch, self.hidden_dim, self.grid_size, self.grid_size)
        z3, z6, z9, z12 = (t.view(shape) for t in (z3, z6, z9, z12))
        z0 = images

        p = self.prompt_encoders["bottleneck"](prompt)
        z12 = self.fusion_layers["bottleneck"](z12, p)

        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        p = self.prompt_encoders["1024"](prompt)
        x = self.fusion_layers["1024"](x, p)
        x = self.c1(x)

        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        p = self.prompt_encoders["768"](prompt)
        x = self.fusion_layers["768"](x, p)
        x = self.c2(x)

        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        p = self.prompt_encoders["512"](prompt)
        x = self.fusion_layers["512"](x, p)
        x = self.c3(x)

        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        p = self.prompt_encoders["128"](prompt)
        x = self.fusion_layers["128"](x, p)
        x = self.c4(x)

        output = self.classifier(x)
        return self.softmax(output)
