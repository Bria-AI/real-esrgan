import numpy as np
import pytest
import torch
import torch.nn.functional as F

from RealESRGAN import RealESRGAN
from RealESRGAN.utils import (
    pad_reflect,
    split_image_into_overlapping_patches,
    stich_together,
    unpad_image,
)


class _Upscale(torch.nn.Module):
    """Deterministic stand-in: bilinear upscale plus a nonlinearity, outputs land between levels."""

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        y = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        return y * 1.1 - 0.03 * torch.sin(7.0 * y)


class _ConvUpscale(torch.nn.Module):
    """Seeded conv stand-in; fp16 output like the TRT engine, and memory layout affects rounding."""

    def __init__(self, scale: int):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 3 * scale * scale, 3, padding=1)
        self.shuffle = torch.nn.PixelShuffle(scale)

    def forward(self, x):
        return torch.sigmoid(self.shuffle(self.conv(x))).half()


_STAND_INS = [_Upscale, _ConvUpscale]


def _make_model(stand_in, scale: int) -> RealESRGAN:
    model = RealESRGAN("cpu", scale=scale)
    model.model = stand_in(scale).eval()
    return model


def _make_frame(h: int, w: int, channels: int, dtype) -> np.ndarray:
    rng = np.random.default_rng(0)
    high = 256 if dtype == np.uint8 else 65536
    return rng.integers(0, high, size=(h, w, channels)).astype(dtype)


def _numpy_predict(
    model, lr_image, batch_size=4, patches_size=192, padding=24, pad_size=15
):
    """Previous numpy implementation of RealESRGAN.predict, kept as the reference."""
    output_dtype = lr_image.dtype
    div_factor = 255 if output_dtype == np.uint8 else 65535
    if lr_image.shape[2] == 4:
        lr_image = lr_image[:, :, :3]
    lr_image = pad_reflect(lr_image, pad_size)
    patches, p_shape = split_image_into_overlapping_patches(
        lr_image, patch_size=patches_size, padding_size=padding
    )
    img = (
        torch.FloatTensor(patches / div_factor)
        .permute((0, 3, 1, 2))
        .to(model.device)
        .detach()
    )
    with torch.no_grad():
        res = model.model(img[0:batch_size])
        for i in range(batch_size, img.shape[0], batch_size):
            res = torch.cat((res, model.model(img[i : i + batch_size])), 0)
    sr_image = res.permute((0, 2, 3, 1)).clamp_(0, 1).cpu()
    np_sr_image = sr_image.numpy()
    padded_size_scaled = tuple(np.multiply(p_shape[0:2], model.scale)) + (3,)
    scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], model.scale)) + (3,)
    np_sr_image = stich_together(
        np_sr_image,
        padded_image_shape=padded_size_scaled,
        target_shape=scaled_image_shape,
        padding_size=padding * model.scale,
    )
    sr_img = (np_sr_image * div_factor).astype(output_dtype)
    return unpad_image(sr_img, pad_size * model.scale)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("stand_in", _STAND_INS)
class TestPredict:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    @pytest.mark.parametrize("scale", [2, 4])
    @pytest.mark.parametrize("channels", [3, 4])
    def test_predict_on_dtype_scale_channels_should_match_numpy_reference(
        self, stand_in, dtype, scale, channels
    ):
        model = _make_model(stand_in, scale)
        frame = _make_frame(100, 130, channels, dtype)

        expected = _numpy_predict(model, frame)
        actual = model.predict(frame)

        assert actual.dtype == expected.dtype
        assert np.array_equal(actual, expected)

    @pytest.mark.parametrize(
        "h,w",
        [
            (162, 162),  # padded size is an exact patch multiple
            (40, 30),  # smaller than one patch
            (200, 70),  # 2x1 tiles from a non-square frame
            (250, 400),  # non-multiple of patch size
        ],
    )
    def test_predict_on_frame_size_should_match_numpy_reference(self, stand_in, h, w):
        model = _make_model(stand_in, 2)
        frame = _make_frame(h, w, 3, np.uint8)

        assert np.array_equal(model.predict(frame), _numpy_predict(model, frame))

    def test_predict_on_batch_size_not_dividing_tile_count_should_match_numpy_reference(
        self, stand_in
    ):
        model = _make_model(stand_in, 2)
        frame = _make_frame(300, 400, 3, np.uint8)

        assert np.array_equal(
            model.predict(frame, batch_size=3),
            _numpy_predict(model, frame, batch_size=3),
        )
