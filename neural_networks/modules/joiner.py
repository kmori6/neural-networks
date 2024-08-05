import torch
import torch.nn as nn


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
