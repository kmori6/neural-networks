import torch
import torch.nn as nn
from neural_networks.modules.conformer import Conformer
from neural_networks.modules.frontend import Frontend
from neural_networks.modules.joiner import Joiner
from neural_networks.modules.predictor import Predictor
from warp_rnnt import rnnt_loss as rnnt_loss_fn


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int,
        d_model: int,
        num_heads: int,
        kernel_size: int,
        num_blocks: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.blank_token_id = vocab_size - 1
        self.ctc_loss_weight = 0.3
        self.frontend = Frontend(n_mels=n_mels)
        self.encoder = Conformer(
            input_size=n_mels,
            d_model=d_model,
            num_heads=num_heads,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
        )
        self.predictor = Predictor(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            blank_token_id=self.blank_token_id,
        )
        self.joiner = Joiner(
            vocab_size=vocab_size,
            encoder_size=d_model,
            predictor_size=hidden_size,
            joiner_size=hidden_size,
            dropout_rate=dropout_rate,
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_token_id, reduction="sum", zero_infinity=True)

    def forward(
        self,
        speech: torch.Tensor,
        speech_length: torch.Tensor,
        token: torch.Tensor,
        target: torch.Tensor,
        target_length: torch.Tensor,
    ):
        x_enc, mask = self.frontend(speech, speech_length)
        x_enc, mask = self.encoder(x_enc, mask[:, None, :])  # (batch, frame, encoder_size)
        x_dec = self.predictor(token)  # (batch, time, predictor_size)
        x = self.joiner(x_enc[:, :, None, :], x_dec[:, None, :, :])  # (batch, frame, time, vocab_size)
        # rnnt loss
        frame_length = mask.squeeze().sum(-1).to(torch.int32)
        rnnt_loss = rnnt_loss_fn(
            x, target, frame_length, target_length, blank=self.blank_token_id, reduction="mean", gather=True
        )
        # ctc
        x_ctc = self.linear(x_enc)  # (batch, frame, vocab_size)
        logp = x_ctc.log_softmax(-1).transpose(0, 1)  # (frame, batch, vocab_size)
        target = target[target != self.blank_token_id]  # (batch * time',)
        ctc_loss = self.ctc_loss_fn(logp, target, frame_length, target_length) / x_ctc.shape[0]
        loss = rnnt_loss + self.ctc_loss_weight * ctc_loss
        return loss, {"loss": loss.item(), "rnnt_loss": rnnt_loss.item(), "ctc_loss": ctc_loss.item()}
