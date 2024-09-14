import torch


def sequence_mask(length: torch.Tensor) -> torch.Tensor:
    return torch.arange(max(length), device=length.device) < length[:, None]


def chunk_mask(length: int, chunk_size: int, num_history_chunks: int) -> torch.Tensor:
    row = torch.arange(length)[:, None]
    col = torch.arange(length)[None, :]
    start = (col // chunk_size) * chunk_size
    return (row >= start) & (col >= start.T - chunk_size * num_history_chunks)


def causal_mask(length: torch.Tensor) -> torch.Tensor:
    max_len = max(length)
    return (
        sequence_mask(length)[:, None, :]
        & torch.ones(1, max_len, max_len, dtype=torch.bool, device=length.device).tril()
    )
