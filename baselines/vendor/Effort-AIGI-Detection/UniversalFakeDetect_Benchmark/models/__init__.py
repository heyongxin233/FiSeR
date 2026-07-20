import os

from .clip_models import ClipModel


VALID_NAMES = {
    "CLIP:ViT-B/16_svd": {
        "local": "/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch16/",
        "hf": "openai/clip-vit-base-patch16",
    },
    "CLIP:ViT-B/32_svd": {
        "local": "/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch32/",
        "hf": "openai/clip-vit-base-patch32",
    },
    "CLIP:ViT-L/14_svd": {
        "local": "/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/",
        "hf": "openai/clip-vit-large-patch14",
    },
}


def resolve_pretrained_name_or_path(name, opt):
    if name not in VALID_NAMES:
        raise ValueError(f"Unsupported backbone name: {name}")

    should_log = (not getattr(opt, "distributed", False)) or getattr(opt, "local_rank", 0) == 0
    explicit_override = getattr(opt, "pretrained_name_or_path", "") or os.environ.get("EFFORT_PRETRAINED_NAME_OR_PATH", "")
    if explicit_override:
        if should_log:
            print(f"Using overridden pretrained backbone: {explicit_override}")
        return explicit_override

    model_entry = VALID_NAMES[name]
    local_path = model_entry.get("local", "")
    if local_path and os.path.exists(local_path):
        if should_log:
            print(f"Using local pretrained backbone: {local_path}")
        return local_path

    hf_name = model_entry.get("hf")
    if hf_name:
        if should_log:
            print(f"Local pretrained backbone not found, fallback to HuggingFace model id: {hf_name}")
        return hf_name

    raise FileNotFoundError(f"No usable pretrained backbone found for {name}")


def get_model(name, opt):
    if name.startswith("CLIP:"):
        return ClipModel(resolve_pretrained_name_or_path(name, opt), opt)
    raise ValueError(f"Unsupported model family in arch: {name}")
