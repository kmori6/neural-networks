import torch


def sequence_mask(length: torch.Tensor) -> torch.Tensor:
    return torch.arange(max(length), device=length.device) < length[:, None]


def chunk_mask(length: int, chunk_size: int, history_window_size: int) -> torch.Tensor:
    """Attention mask design for training streaming models

    Reference: https://arxiv.org/abs/2010.11395

    """
    if chunk_size < 0:
        return torch.ones([length, length], dtype=torch.bool)
    row = torch.arange(length)[:, None]
    col = torch.arange(length)[None, :]
    row_start = (col // chunk_size) * chunk_size
    col_start = row_start.T - history_window_size
    return (row >= row_start) & (col >= col_start)


def causal_mask(length: torch.Tensor) -> torch.Tensor:
    max_len = max(length)
    return (
        sequence_mask(length)[:, None, :]
        & torch.ones(1, max_len, max_len, dtype=torch.bool, device=length.device).tril()
    )
