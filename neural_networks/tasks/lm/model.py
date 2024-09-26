import math

import torch
import torch.nn as nn

from neural_networks.modules.transformer import AbsolutePositionalEncoding, Encoder
from neural_networks.utils.mask import causal_mask


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout_rate: float,
        pad_token_id: int,
        bos_token_id: int = 1,
        eos_token_id: int = 1,
        label_smoothing: float = 0.1,
        ignore_token_id: int = -100,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.ignore_token_id = ignore_token_id
        self.scale = math.sqrt(d_model)
        self.input_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = AbsolutePositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout_rate)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
        self.linear.weight = self.input_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    def forward(
        self, token: torch.Tensor, token_length: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # encoder
        mask_enc = causal_mask(token_length)[:, None, :, :]  # (batch, 1, time, time)
        x = self.input_embedding(token)  # (batch, time, d_model)
        x = x * self.scale + self.positional_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, mask_enc)
        x = self.linear(x)  # (batch, time, vocab_size)
        # loss
        loss = self.loss_fn(x.flatten(0, 1), target.flatten())
        mask_valid = target != self.ignore_token_id
        acc = (x.argmax(-1)[mask_valid] == target[mask_valid]).sum() / mask_valid.sum()
        pp = torch.exp(loss)
        return loss, {"loss": loss.item(), "acc": acc.item(), "pp": pp.item()}
