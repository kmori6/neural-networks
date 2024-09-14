from typing import Any

import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 640,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        blank_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=blank_token_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device)
        return h, c

    def forward(
        self, token: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """

        Args:
            token (torch.Tensor): Input tensor (batch, time).

        Returns:
            torch.Tensor: Output tensor (batch, time, hidden_size).
        """
        x = self.embed(token)  # (batch, time, hidden_size)
        x = self.dropout(x)
        x, (h, c) = self.lstm(x, state)
        return x, (h, c)


class Joiner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_size: int = 512,
        predictor_size: int = 640,
        joiner_size: int = 640,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.linear_enc = nn.Linear(encoder_size, joiner_size)
        self.linear_pred = nn.Linear(predictor_size, joiner_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_out = nn.Linear(joiner_size, vocab_size)

    def forward(self, x_enc: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder output sequence (batch, frame, 1, encoder_size).
            x_pred (torch.Tensor): Predictor output sequence (batch, 1, time, predictor_size).

        Returns:
            torch.Tensor: Output log-probability sequence (batch, frame, time, vocab_size).
        """
        x = self.linear_enc(x_enc) + self.linear_pred(x_pred)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        x = x.log_softmax(dim=-1)
        return x


def default_beam_search(
    x_enc: torch.Tensor,
    beam_size: int,
    B: list[dict[str, Any]],
    cache: dict[tuple, Any],
    predictor: Predictor,
    joiner: Joiner,
) -> tuple[list[dict[str, Any]], dict[tuple, Any]]:
    """Default beam search in https://arxiv.org/abs/2201.05420 (section 5.1).

    Args:
        x_enc (torch.Tensor): Encoder sequence (time, d_model).
        beam_size (int): Beam witch.
        B (list[dict[str, Any]]): Current hypothesis.
        cache (dict[tuple, Any]): Prediction cache.
        predictor (Predictor): Predictor.
        joiner (Joiner): Joiner.

    Returns:
        tuple[list[dict[str, Any]], dict[tuple, Any]]:
            tuple[list[dict[str, Any]]: Complete hypothesis list
            dict[tuple, Any]: Prediction cache
    """
    for t in range(x_enc.shape[0]):
        A = B
        B = []
        while True:
            best_hyp = max(A, key=lambda x: x["score"])
            A.remove(best_hyp)
            key = tuple(best_hyp["token"])
            if tuple(key) in cache:
                x_pred, state = cache[key]
            else:
                x_pred, state = predictor(
                    torch.tensor([best_hyp["token"][-1:]], dtype=torch.long, device=x_enc.device),
                    best_hyp["state"],
                )
                cache[key] = x_pred, state
            logp = joiner(x_enc[None, t : t + 1, :], x_pred).squeeze()  # (vocab_size,)
            topk = logp[:-1].topk(beam_size)
            B.append(
                {"score": best_hyp["score"] + logp[-1].item(), "token": best_hyp["token"], "state": best_hyp["state"]}
            )
            for score, k in zip(*topk):
                A.append(
                    {"score": best_hyp["score"] + score.item(), "token": best_hyp["token"] + [int(k)], "state": state}
                )
            _B = sorted(
                [hyp for hyp in B if hyp["score"] > max(A, key=lambda x: x["score"])["score"]], key=lambda x: x["score"]
            )
            if len(_B) >= beam_size:
                B = _B
                break

    return sorted(B, key=lambda x: x["score"] / len(x["token"]), reverse=True), cache
