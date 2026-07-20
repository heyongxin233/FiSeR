from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel


class FiSeREncoder(nn.Module):
    def __init__(self, model_name_or_path: str, local_files_only: bool = False) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def enable_gradient_checkpointing(self) -> None:
        method = getattr(self.encoder, "gradient_checkpointing_enable", None)
        if callable(method):
            try:
                method(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                method()

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        output = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if output_hidden_states:
            assert output.hidden_states is not None
            return tuple(state[:, 0, :] for state in output.hidden_states)
        return output.last_hidden_state[:, 0, :]

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        self.encoder.save_pretrained(output_dir, **kwargs)


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "encoder"):
        value = payload.get(key)
        if isinstance(value, dict):
            payload = value
            break

    result: dict[str, torch.Tensor] = {}
    for raw_key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        for prefix in ("encoder.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key in {"vitual_center", "virtual_center"}:
            continue
        result[key] = value
    return result


def load_encoder(
    model_name_or_path: str,
    checkpoint: str | None = None,
    local_files_only: bool = False,
    strict: bool = True,
) -> FiSeREncoder:
    """Load a Hugging Face model directory/repo or a legacy FiSeR `.pth` file."""
    model = FiSeREncoder(model_name_or_path, local_files_only=local_files_only)
    if not checkpoint:
        return model

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        return FiSeREncoder(str(checkpoint_path), local_files_only=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(payload)
    incompatible = model.encoder.load_state_dict(state_dict, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            f"Checkpoint loaded with missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    return model
