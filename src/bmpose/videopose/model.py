from __future__ import annotations

import torch
import torch.nn as nn


class TemporalModelBase(nn.Module):
    """A minimal VideoPose3D-compatible temporal model base."""

    def __init__(
        self,
        num_joints_in: int,
        in_features: int,
        num_joints_out: int,
        filter_widths: list[int] | tuple[int, ...],
        causal: bool,
        dropout: float,
        channels: int,
    ) -> None:
        super().__init__()
        for filter_width in filter_widths:
            if filter_width % 2 == 0:
                raise ValueError("Only odd filter widths are supported.")

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = list(filter_widths)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.pad = [self.filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def receptive_field(self) -> int:
        return 1 + (2 * sum(self.pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input shape [batch, time, joints, features].")
        if x.shape[-2] != self.num_joints_in or x.shape[-1] != self.in_features:
            raise ValueError("Unexpected joint or feature dimension.")

        batch_size, _, _, _ = x.shape
        x = x.view(batch_size, x.shape[1], -1).permute(0, 2, 1)
        x = self._forward_blocks(x)
        x = x.permute(0, 2, 1)
        return x.view(batch_size, -1, self.num_joints_out, 3)


class TemporalModel(TemporalModelBase):
    """VideoPose3D-compatible temporal convolution network for inference."""

    def __init__(
        self,
        num_joints_in: int,
        in_features: int,
        num_joints_out: int,
        filter_widths: list[int] | tuple[int, ...],
        causal: bool = False,
        dropout: float = 0.25,
        channels: int = 1024,
        dense: bool = False,
    ) -> None:
        super().__init__(
            num_joints_in=num_joints_in,
            in_features=in_features,
            num_joints_out=num_joints_out,
            filter_widths=filter_widths,
            causal=causal,
            dropout=dropout,
            channels=channels,
        )
        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], bias=False)

        layers_conv: list[nn.Module] = []
        layers_bn: list[nn.Module] = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]

        for filter_width in filter_widths[1:]:
            self.pad.append(((filter_width - 1) * next_dilation) // 2)
            self.causal_shift.append(((filter_width // 2) * next_dilation) if causal else 0)

            effective_width = filter_width if not dense else (2 * self.pad[-1] + 1)
            dilation = next_dilation if not dense else 1
            layers_conv.append(nn.Conv1d(channels, channels, effective_width, dilation=dilation, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_width

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for block_index in range(len(self.pad) - 1):
            pad = self.pad[block_index + 1]
            shift = self.causal_shift[block_index + 1]
            residual = x[:, :, pad + shift : x.shape[2] - pad + shift]
            x = self.drop(self.relu(self.layers_bn[2 * block_index](self.layers_conv[2 * block_index](x))))
            x = residual + self.drop(
                self.relu(self.layers_bn[(2 * block_index) + 1](self.layers_conv[(2 * block_index) + 1](x)))
            )

        return self.shrink(x)
