import math

import torch
import torch.nn as nn

from neural_networks.modules.transformer import Decoder, Encoder, PositionalEncoding
from neural_networks.utils.mask import causal_mask, sequence_mask


class Model(nn.Module):
    """Transformer encoder-decoder architecture

    Reference: https://arxiv.org/abs/1706.03762

    """

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
        self.output_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout_rate)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
        self.output_embedding = self.input_embedding
        self.linear.weight = self.input_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)

    def forward(
        self,
        enc_token: torch.Tensor,
        enc_token_length: torch.Tensor,
        dec_token: torch.Tensor,
        dec_token_length: torch.Tensor,
        tgt_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """

        Args:
            enc_token (torch.Tensor): Encoder input token sequence (batch, time1).
            enc_token_length (torch.Tensor): Encoder input token length (batch,).
            dec_token (torch.Tensor): Decoder input token sequence (batch, time2).
            dec_token_length (torch.Tensor): Decoder input token length (batch,).
            tgt_token (torch.Tensor): Target token sequence (batch, time2).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - torch.Tensor: Loss value.
                - dict[str, torch.Tensor]: Training statistics dictionary.
        """
        # encoder
        mask_enc = sequence_mask(enc_token_length)[:, None, None, :]  # (batch, 1, 1, time1)
        x_enc = self.input_embedding(enc_token)  # (batch, time1, d_model)
        x_enc = x_enc * self.scale + self.positional_encoding(x_enc.shape[1], x_enc.dtype, x_enc.device)
        x_enc = self.dropout1(x_enc)
        x_enc = self.encoder(x_enc, mask_enc)
        # decoder
        x_dec = self.output_embedding(dec_token)  # (batch, time2, d_model)
        x_dec = x_dec * self.scale + self.positional_encoding(x_dec.shape[1], x_dec.dtype, x_dec.device)
        x_dec = self.dropout2(x_dec)
        mask_dec = causal_mask(dec_token_length)[:, None, :, :]  # (batch, 1, time2, time2)
        x_dec = self.decoder(x_enc, x_dec, mask_enc, mask_dec)
        x_dec = self.linear(x_dec)  # (batch, time2, vocab_size)
        # loss
        loss = self.loss_fn(x_dec.flatten(0, 1), tgt_token.flatten()) / x_dec.shape[0]
        mask_valid = tgt_token != self.ignore_token_id
        acc = (x_dec.argmax(-1)[mask_valid] == tgt_token[mask_valid]).sum() / mask_valid.sum()
        return loss, {"loss": loss.item(), "acc": acc.item()}
