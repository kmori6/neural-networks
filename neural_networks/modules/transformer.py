import math

import torch
import torch.nn as nn


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.init_pe(max_length)

    def init_pe(self, max_length: int):
        t = torch.arange(max_length, dtype=torch.float32)
        omega = torch.exp(-torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model * math.log(10000.0))
        theta = omega[None, :] * t[:, None]  # (max_length, d_model / 2)
        pe = torch.zeros(max_length, self.d_model)
        pe[:, 0::2] = torch.sin(theta)
        pe[:, 1::2] = torch.cos(theta)
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, time, d_model).

        Returns:
            torch.Tensor: Position embedding (batch, time, d_model).
        """
        return self.pe[:, : x.shape[1]]


class RelativePositionalEncoding(AbsolutePositionalEncoding):
    def __init__(self, d_model: int, max_length: int = 4096):
        super().__init__(d_model, max_length)
        self.d_model = d_model
        self.init_pe(max_length)

    def init_pe(self, max_length: int):
        t = torch.arange(max_length - 1, -max_length, -1, dtype=torch.float32)
        omega = torch.exp(-torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model * math.log(10000.0))
        theta = omega[None, :] * t[:, None]  # (2 * max_length - 1, d_model / 2)
        pe = torch.zeros(2 * max_length - 1, self.d_model)
        pe[:, 0::2] = torch.sin(theta)
        pe[:, 1::2] = torch.cos(theta)
        self.register_buffer("pe", pe[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, time, d_model).

        Returns:
            torch.Tensor: Position embedding (batch, 2 * time - 1, d_model).
        """
        base_position = self.pe.shape[1] // 2
        pe = self.pe[:, base_position - x.shape[1] + 1 : base_position + x.shape[1]]
        return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query sequence (batch, time1, d_model).
            k (torch.Tensor): Key sequence (batch, time2, d_model).
            v (torch.Tensor): Value sequence (batch, time2, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1 or time1, time2).

        Returns:
            torch.Tensor: Output sequence (batch, time1, d_model).
        """
        b = q.shape[0]
        q = self.w_q(q).view(b, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time1, d_k)
        k = self.w_k(k).view(b, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)
        v = self.w_v(v).view(b, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)
        x = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        x = x.masked_fill(~mask, float("-inf"))
        x = torch.softmax(x, dim=-1)
        x = self.dropout(x)
        x = torch.matmul(x, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).flatten(2, 3)  # (batch, time1, d_model)
        x = self.w_o(x)
        return x


class RelativePositionalMultiHeadAttention(MultiHeadAttention):
    """multi-head attention with relative positional embedding described in https://arxiv.org/abs/1901.02860."""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__(d_model, num_heads, dropout_rate)
        self.w_p = nn.Linear(d_model, d_model, bias=False)
        self.b_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.b_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.b_u)
        nn.init.xavier_uniform_(self.b_v)

    def forward(  # type: ignore[override]
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, p: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query sequence (batch, time1, d_model).
            k (torch.Tensor): Key sequence (batch, time2, d_model).
            v (torch.Tensor): Value sequence (batch, time2, d_model).
            p (torch.Tensor): Position embedding (batch, 2 * time2 - 1, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1 or time1, time2).

        Returns:
            torch.Tensor: Output sequence (batch, time1, d_model).
        """
        t1, t2 = q.shape[1], k.shape[1]
        assert p.shape[1] >= 2 * t2 - 1
        q = self.w_q(q).view(-1, t1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time1, d_k)
        k = self.w_k(k).view(-1, t2, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)
        v = self.w_v(v).view(-1, t2, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)
        p = self.w_p(p).view(1, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, 2 * time1 - 1, d_k)
        # attention score in section 3.3
        ac = torch.matmul(q + self.b_u[None, :, None, :], k.transpose(2, 3))  # (batch, head, time1, time2)
        # efficient computation in appendix B
        bd = torch.matmul(q + self.b_v[None, :, None, :], p.transpose(2, 3))  # (batch, head, time1, 2 * time2 - 1)
        bd = bd.flatten(2, 3)[..., t1 - 1 :].unfold(-1, t2, bd.shape[-1] - 1)  # (batch, head, time1, time2)
        x = (ac + bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        x = x.masked_fill(~mask, float("-inf"))
        x = torch.softmax(x, dim=-1)
        x = self.dropout(x)
        x = torch.matmul(x, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).flatten(2, 3)  # (batch, time1, d_model)
        x = self.w_o(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, act_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act_fn = act_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, time, d_model).

        Returns:
            torch.Tensor: Output sequence (batch, time, d_model).
        """
        x = self.w_1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, time, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1, time).

        Returns:
            torch.Tensor: Output sequence (batch, time, d_model).
        """
        x = self.layernorm1(x + self.dropout1(self.mha(x, x, x, mask)))
        x = self.layernorm2(x + self.dropout2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, mask_enc: torch.Tensor, mask_dec: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder sequence (batch, time1, d_model).
            x_dec (torch.Tensor): Decoder sequence (batch, time2, d_model).
            mask_enc (torch.Tensor): Encoder mask (batch, 1, 1, time1).
            mask_dec (torch.Tensor): Decoder mask (batch, 1, time2, time2).

        Returns:
            torch.Tensor: Output sequence (batch, time2, d_model).
        """
        x_dec = self.layernorm1(x_dec + self.dropout1(self.masked_mha(x_dec, x_dec, x_dec, mask_dec)))
        x_dec = self.layernorm2(x_dec + self.dropout2(self.mha(x_dec, x_enc, x_enc, mask_enc)))
        x_dec = self.layernorm3(x_dec + self.dropout3(self.ffn(x_dec)))
        return x_dec


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout_rate: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input sequence (batch, time, d_model).
            mask (torch.Tensor): Mask (batch, 1, 1, time).

        Returns:
            torch.Tensor: Output sequence (batch, time, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout_rate: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])

    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, mask_enc: torch.Tensor, mask_dec: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder sequence (batch, time1, d_model).
            x_dec (torch.Tensor): Decoder sequence (batch, time2, d_model).
            mask_enc (torch.Tensor): Encoder mask (batch, 1, 1, time1).
            mask_dec (torch.Tensor): Decoder mask (batch, 1, time2, time2).

        Returns:
            torch.Tensor: Output sequence (batch, time2, d_model).
        """
        for layer in self.layers:
            x_dec = layer(x_enc, x_dec, mask_enc, mask_dec)
        return x_dec
