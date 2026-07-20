import math
from io import BytesIO
from typing import Optional

import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import InterpolationMode


def normalize_corruption_type(corruption_type: Optional[str]) -> str:
    corruption_type = str(corruption_type or "none").strip().lower()
    alias = {
        "": "none",
        "none": "none",
        "clean": "none",
        "gaussian": "gaussian_blur",
        "blur": "gaussian_blur",
        "gaussian_blur": "gaussian_blur",
        "crop": "crop",
        "center_crop": "crop",
        "jpeg": "jpeg",
        "jpeg_compression": "jpeg",
    }
    if corruption_type not in alias:
        raise ValueError(
            f"Unsupported corruption_type='{corruption_type}', "
            "expected one of: none, gaussian_blur, jpeg, crop"
        )
    return alias[corruption_type]


def format_corruption_value(corruption_value: float) -> str:
    value = float(corruption_value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def apply_pil_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB")
    out.load()
    return out


def apply_image_corruption(
    image: Image.Image,
    corruption_type: str = "none",
    corruption_value: float = 0.0,
    crop_mode: str = "center",
) -> Image.Image:
    corruption_type = normalize_corruption_type(corruption_type)
    if not isinstance(image, Image.Image):
        return image

    image = image.convert("RGB")
    value = float(corruption_value)
    if corruption_type == "none" or value <= 0:
        return image

    if corruption_type == "jpeg":
        quality = max(1, min(100, int(round(value))))
        return apply_pil_jpeg(image, quality)

    if corruption_type == "crop":
        if str(crop_mode).strip().lower() != "center":
            raise ValueError(f"Unsupported crop_mode='{crop_mode}', only 'center' is supported for now.")
        width, height = image.size
        area_ratio = min(1.0, max(1e-6, value))
        side_ratio = math.sqrt(area_ratio)
        crop_w = max(1, min(width, int(round(width * side_ratio))))
        crop_h = max(1, min(height, int(round(height * side_ratio))))
        cropped = TVF.center_crop(image, output_size=[crop_h, crop_w])
        return TVF.resize(
            cropped,
            size=[height, width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    sigma = float(value)
    kernel_size = max(3, 1 + 2 * round(sigma * 4.0))
    return TVF.gaussian_blur(image, kernel_size=kernel_size, sigma=[sigma, sigma])
