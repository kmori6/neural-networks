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
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=blank_token_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        """

        Args:
            token (torch.Tensor): Input tensor (batch, time).

        Returns:
            torch.Tensor: Output tensor (batch, time, hidden_size).
        """
        x = self.embed(token)  # (batch, time, hidden_size)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return x
