from __future__ import annotations

import inspect

from monai.networks.nets import UNet


def build_model_with_name(
    model_name: str,
    in_channels: int,
    out_channels: int,
    patch_size: tuple[int, int, int] = (128, 128, 128),
) -> object:
    name = model_name.lower().strip()
    if name == "unet":
        return build_model(in_channels=in_channels, out_channels=out_channels)

    if name == "swinunetr":
        try:
            from monai.networks.nets import SwinUNETR
        except Exception as exc:
            raise RuntimeError("SwinUNETR is not available in this MONAI installation") from exc

        init_params = inspect.signature(SwinUNETR.__init__).parameters
        kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
        }
        if "img_size" in init_params:
            kwargs["img_size"] = patch_size
        if "feature_size" in init_params:
            kwargs["feature_size"] = 24
        if "use_checkpoint" in init_params:
            kwargs["use_checkpoint"] = False
        return SwinUNETR(**kwargs)

    raise ValueError(f"Unsupported model_name='{model_name}'. Use 'unet' or 'swinunetr'.")


def build_model(in_channels: int = 3, out_channels: int = 2) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
