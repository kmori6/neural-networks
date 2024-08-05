import torch


def sequence_mask(length: torch.Tensor) -> torch.Tensor:
    return torch.arange(max(length), device=length.device) < length[:, None]
