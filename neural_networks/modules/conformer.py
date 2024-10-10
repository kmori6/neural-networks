from typing import Optional

import torch
import torch.nn as nn

from neural_networks.modules.transformer import (
    FeedForward,
    RelativePositionalMultiHeadAttention,
)
from neural_networks.utils.mask import chunk_mask, sequence_mask


class ConvolutionSubsampling(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, output_size, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=2)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, frame, n_mel).
            mask (torch.Tensor): Mask (batch, frame).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Output sequence (batch, frame', d_model).
                torch.Tensor: Mask (batch, frame').
        """
        x = x[:, None, :, :]  # (batch, 1, frame, feat_size)
        x = self.conv1(x)  # (batch, output_size, frame, feat_size)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.transpose(1, 2).flatten(2, -1)  # (batch, frame, output_size * feat_size')
        return x, sequence_mask(((mask.sum(-1) - 1) // 2 - 1) // 2)


class FeedForwardModule(FeedForward):
    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__(d_model, 4 * d_model, dropout_rate, act_fn=nn.SiLU())
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, frame, d_model).

        Returns:
            torch.Tensor: Output sequence (batch, frame, d_model).
        """
        x = self.layernorm(x)
        x = super().forward(x)
        x = self.dropout2(x)
        return x


class MultiHeadSelfAttentionModule(RelativePositionalMultiHeadAttention):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__(d_model, num_heads, dropout_rate)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """

        Args:
            x (torch.Tensor): Input sequence (batch, frame, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1, frame).

        Returns:
            torch.Tensor: Output sequence (batch, frame, d_model).
        """
        x = self.layernorm(x)
        x = super().forward(x, x, x, mask)
        x = self.dropout2(x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0)
        self.glu_activation = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=d_model
        )
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.swish_activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, frame, d_model).

        Returns:
            torch.Tensor: Output sequence (batch, frame, d_model).
        """
        x = self.layernorm(x)
        x = x.transpose(1, 2)  # (batch, d_model, frame)
        x = self.pointwise_conv1(x)  # (batch, 2 * d_model, frame)
        x = self.glu_activation(x)  # (batch, d_model, frame)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.swish_activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # (batch, frame, d_model)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout_rate)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, num_heads, dropout_rate)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout_rate)
        self.ffn2 = FeedForwardModule(d_model, dropout_rate)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, frame, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1, frame).

        Returns:
            torch.Tensor: Output sequence (batch, frame, d_model).
        """
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, mask)
        x = x + self.conv(x)
        x = self.layernorm(x + 0.5 * self.ffn2(x))
        return x


class Conformer(nn.Module):
    def __init__(
        self,
        input_size: int = 80,
        d_model: int = 512,
        num_heads: int = 8,
        kernel_size: int = 31,
        num_blocks: int = 17,
        dropout_rate: float = 0.1,
        chunk_size: Optional[int] = None,
        num_history_chunks: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.num_history_chunks = num_history_chunks
        self.convolution_subsampling = ConvolutionSubsampling(d_model)
        self.linear = nn.Linear(d_model * (((input_size - 1) // 2 - 1) // 2), d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.conformer_blocks = nn.ModuleList(
            [ConformerBlock(d_model, num_heads, kernel_size, dropout_rate) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, frame, input_size).
            mask (torch.Tensor): Mask (batch, frame).

        Returns:
            torch.Tensor: Output sequence (batch, frame', d_model).
        """
        x, mask = self.convolution_subsampling(x, mask)
        x = self.linear(x)
        x = self.dropout(x)
        if self.training and self.chunk_size and self.num_history_chunks:
            attn_mask = chunk_mask(x.shape[1], self.chunk_size, self.num_history_chunks).to(mask.device)[
                None, None, :, :
            ]  # (1, 1, frame', frame')
        else:
            attn_mask = mask[:, None, None, :]  # (batch, 1, 1, frame')
        for block in self.conformer_blocks:
            x = block(x, attn_mask)
        return x, mask
