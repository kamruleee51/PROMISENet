import torch
import torch.nn as nn
import torch.nn.functional as F

from promisenet.models.pcca import BoxPositionEncoder, PromptSEFusion_gate


class ConvolutionalBlock(nn.Module):
    """Two conv layers, each followed by batch norm and ReLU."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.batch_norm1(self.conv1(x)))
        x = self.activation(self.batch_norm2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Conv block followed by max pooling."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv_block = ConvolutionalBlock(input_channels, output_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feature_map = self.conv_block(x)
        pooled_feature_map = self.pooling(feature_map)
        return feature_map, pooled_feature_map


class DecoderBlock(nn.Module):
    """Upsample + skip-concat + conv, then PCCA gate — the decoder-stage re-application of PCCA."""

    def __init__(self, in_channels: int, out_channels: int, prompt_channels: int):
        super().__init__()
        self.out_channels = out_channels

        self.upsample_x = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.upsample_prompt = nn.ConvTranspose2d(prompt_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvolutionalBlock(out_channels * 2, out_channels)
        self.fusion_layers = PromptSEFusion_gate(out_channels, out_channels, reduction=16)

    def forward(self, x, skip_connection, prompt):
        x = self.upsample_x(x)
        prompt = self.upsample_prompt(prompt)

        target_size = x.shape[-2:]
        if skip_connection.shape[-2:] != target_size:
            skip_connection = F.interpolate(skip_connection, size=target_size, mode="bilinear", align_corners=False)
        if prompt.shape[-2:] != target_size:
            prompt = F.interpolate(prompt, size=target_size, mode="bilinear", align_corners=False)

        concat = torch.cat([x, skip_connection], dim=1)
        concat_conv = self.conv_block(concat)
        return self.fusion_layers(concat_conv, prompt)


class PROMISENet_CNN(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 3,
        num_classes: int = 2,
        feature_sizes=(64, 128, 256, 512, 1024),
        prompt_sizes=(64, 128, 256, 512),
        image_size=(256, 256),
    ):
        super().__init__()
        feature_sizes = list(feature_sizes)
        prompt_sizes = list(prompt_sizes)

        self.encoder1 = EncoderBlock(num_input_channels, feature_sizes[0])
        self.encoder2 = EncoderBlock(feature_sizes[0], feature_sizes[1])
        self.encoder3 = EncoderBlock(feature_sizes[1], feature_sizes[2])
        self.encoder4 = EncoderBlock(feature_sizes[2], feature_sizes[3])
        self.bottleneck = ConvolutionalBlock(feature_sizes[3], feature_sizes[4])

        # Encoder-stage + bottleneck prompt encoders/gates.
        # Spatial sizes below assume a 256x256 input (4 successive /2 poolings: 128,64,32,16).
        h, w = image_size
        self.prompt_encoders = nn.ModuleDict({
            "128": BoxPositionEncoder(feature_sizes[0], image_size, (h // 2, w // 2)),
            "64": BoxPositionEncoder(feature_sizes[1], image_size, (h // 4, w // 4)),
            "32": BoxPositionEncoder(feature_sizes[2], image_size, (h // 8, w // 8)),
            "16": BoxPositionEncoder(feature_sizes[3], image_size, (h // 16, w // 16)),
            "bottleneck": BoxPositionEncoder(feature_sizes[4], image_size, (h // 16, w // 16)),
        })
        self.fusion_layers = nn.ModuleDict({
            "128": PromptSEFusion_gate(feature_sizes[0], feature_sizes[0], reduction=16),
            "64": PromptSEFusion_gate(feature_sizes[1], feature_sizes[1], reduction=16),
            "32": PromptSEFusion_gate(feature_sizes[2], feature_sizes[2], reduction=16),
            "16": PromptSEFusion_gate(feature_sizes[3], feature_sizes[3], reduction=16),
            "bottleneck": PromptSEFusion_gate(feature_sizes[4], feature_sizes[4], reduction=16),
        })

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                in_channels=feature_sizes[i + 1],
                out_channels=feature_sizes[i],
                prompt_channels=prompt_sizes[i],
            )
            for i in range(len(feature_sizes) - 1)
        ])

        self.classifier = nn.Conv2d(feature_sizes[0], num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W] input image.
            prompt: [B, N, 4] bounding box(es) in image pixel coordinates.
        Returns:
            [B, num_classes, H, W] softmax probabilities (softmax applied internally — this model
            returns probabilities, not raw logits, unlike PROMISE-Txformer).
        """
        skip1, p1 = self.encoder1(x)
        p1_prompt = self.prompt_encoders["128"](prompt)
        p1 = self.fusion_layers["128"](p1, p1_prompt)

        skip2, p2 = self.encoder2(p1)
        p2_prompt = self.prompt_encoders["64"](prompt)
        p2 = self.fusion_layers["64"](p2, p2_prompt)

        skip3, p3 = self.encoder3(p2)
        p3_prompt = self.prompt_encoders["32"](prompt)
        p3 = self.fusion_layers["32"](p3, p3_prompt)

        skip4, p4 = self.encoder4(p3)
        p4_prompt = self.prompt_encoders["16"](prompt)
        p4 = self.fusion_layers["16"](p4, p4_prompt)

        bottleneck = self.bottleneck(p4)
        bottleneck_prompt = self.prompt_encoders["bottleneck"](prompt)
        bottleneck = self.fusion_layers["bottleneck"](bottleneck, bottleneck_prompt)

        decoded = self.decoder_blocks[3](bottleneck, skip4, p4_prompt)
        decoded = self.decoder_blocks[2](decoded, skip3, p3_prompt)
        decoded = self.decoder_blocks[1](decoded, skip2, p2_prompt)
        decoded = self.decoder_blocks[0](decoded, skip1, p1_prompt)

        segmentation_mask = self.classifier(decoded)
        return self.softmax(segmentation_mask)
