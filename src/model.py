from __future__ import annotations

from monai.networks.nets import UNet


def build_model(in_channels: int = 3, out_channels: int = 2) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
