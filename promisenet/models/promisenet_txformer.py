"""
PROMISE-Txformer: MONAI UNETR backbone with PCCA applied at each encoder-stage projection and the
ViT bottleneck.

Faithful extraction from Unetr_monai/gated_unetr/{dataset}/*.ipynb — confirmed as the exact source of
every PROMISE-Txformer number reported in the paper's Table TAM_SAM (16/16 dataset x metric
combinations matched the paper to the reported decimal; see the audit in the source repo's
agent_files/09_txformer_decoder_discrepancy.md).

IMPORTANT — known, confirmed paper-vs-code gap, preserved intentionally:
PCCA here is applied only at the 3 encoder-stage projections (enc2/enc3/enc4) and the ViT bottleneck.
It is NOT re-applied inside decoder5/decoder4/decoder3/decoder2, unlike PROMISE-CNN (which does gate
every decoder stage). The paper's decoder equation (eq:decoder_update_pcca) describes decoder-stage
gating for both variants, but the model that actually produced the reported PROMISE-Txformer numbers
does not do this for the transformer variant. This is shipped as-is (matching the reported results) —
see agent_files/09_txformer_decoder_discrepancy.md for the full evidence and the paper-text correction
that accompanies this choice.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep

from promisenet.models.pcca import BoxPositionEncoder, PromptSEFusion_gate


class PROMISENet_txformer(nn.Module):
    """UNETR (Hatamizadeh et al.) + PCCA. All four datasets use identical hyperparameters below
    (verified against the source notebooks); only in_channels/out_channels vary."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        img_size: Sequence[int] | int = (256, 256),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "perceptron",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=feature_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims, in_channels=hidden_size, out_channels=feature_size * 2,
            num_layer=2, kernel_size=3, stride=1, upsample_kernel_size=2,
            norm_name=norm_name, conv_block=conv_block, res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims, in_channels=hidden_size, out_channels=feature_size * 4,
            num_layer=1, kernel_size=3, stride=1, upsample_kernel_size=2,
            norm_name=norm_name, conv_block=conv_block, res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims, in_channels=hidden_size, out_channels=feature_size * 8,
            num_layer=0, kernel_size=3, stride=1, upsample_kernel_size=2,
            norm_name=norm_name, conv_block=conv_block, res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=hidden_size, out_channels=feature_size * 8,
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=feature_size * 8, out_channels=feature_size * 4,
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=feature_size * 4, out_channels=feature_size * 2,
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=feature_size * 2, out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

        # PCCA at each encoder-stage projection + bottleneck (see module docstring: NOT re-applied
        # in the decoder). Spatial sizes below assume a 256x256 input with patch_size=16 (ViT grid
        # 16x16); MONAI's UnetrPrUpBlock upsamples by 2^(num_layer+1) from that grid.
        self.prompt_encoders = nn.ModuleDict({
            "32": BoxPositionEncoder(feature_size * 2, img_size, (self.feat_size[0] * 8, self.feat_size[1] * 8)),
            "64": BoxPositionEncoder(feature_size * 4, img_size, (self.feat_size[0] * 4, self.feat_size[1] * 4)),
            "128": BoxPositionEncoder(feature_size * 8, img_size, (self.feat_size[0] * 2, self.feat_size[1] * 2)),
            "768": BoxPositionEncoder(hidden_size, img_size, self.feat_size),
        })
        self.fusion_layers = nn.ModuleDict({
            "32": PromptSEFusion_gate(feature_size * 2, feature_size * 2, reduction=16),
            "64": PromptSEFusion_gate(feature_size * 4, feature_size * 4, reduction=16),
            "128": PromptSEFusion_gate(feature_size * 8, feature_size * 8, reduction=16),
            "768": PromptSEFusion_gate(hidden_size, hidden_size, reduction=16),
        })

    def proj_feat(self, x: torch.Tensor) -> torch.Tensor:
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        return x.permute(self.proj_axes).contiguous()

    def forward(self, x_in: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_in: [B, C_in, H, W] input image.
            prompt: [B, N, 4] bounding box(es) in image pixel coordinates.
        Returns:
            [B, num_classes, H, W] raw logits (no softmax applied — unlike PROMISE-CNN, which does
            apply softmax internally). Callers must apply softmax/cross-entropy accordingly.
        """
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)

        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        p = self.prompt_encoders["32"](prompt)
        enc2 = self.fusion_layers["32"](enc2, p)

        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        p = self.prompt_encoders["64"](prompt)
        enc3 = self.fusion_layers["64"](enc3, p)

        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        p = self.prompt_encoders["128"](prompt)
        enc4 = self.fusion_layers["128"](enc4, p)

        dec4 = self.proj_feat(x)
        p = self.prompt_encoders["768"](prompt)
        dec4 = self.fusion_layers["768"](dec4, p)

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        return self.out(out)
