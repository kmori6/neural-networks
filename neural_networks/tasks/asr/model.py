import torch
import torch.nn as nn
from torchaudio.transforms import RNNTLoss

from neural_networks.modules.conformer import Conformer
from neural_networks.modules.frontend import Frontend
from neural_networks.modules.transducer import Joiner, Predictor


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
        ctc_loss_weight: float,
        chunk_size: int,
        history_window_size: int,
    ):
        super().__init__()
        self.blank_token_id = vocab_size - 1
        self.ctc_loss_weight = ctc_loss_weight
        self.frontend = Frontend(n_mels=n_mels)
        self.encoder = Conformer(
            input_size=n_mels,
            d_model=d_model,
            num_heads=num_heads,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            chunk_size=chunk_size,
            history_window_size=history_window_size,
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
        self.rnnt_loss = RNNTLoss(blank=self.blank_token_id, reduction="mean", fused_log_softmax=False)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_token_id, reduction="sum", zero_infinity=True)

    def forward(
        self,
        audio: torch.Tensor,
        audio_length: torch.Tensor,
        token: torch.Tensor,
        target: torch.Tensor,
        target_length: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """

        Args:
            audio (torch.Tensor): Sqeech (batch, sample).
            audio_length (torch.Tensor): Speech length (batch,).
            token (torch.Tensor): Predictor input token (batch, length + 1).
            target (torch.Tensor): Target token (batch, length).
            target_length (torch.Tensor): Target token length (batch,).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor: Loss.
                dict[str, torch.Tensor]: Statistics.
        """
        b = audio.shape[0]
        x_enc, mask = self.frontend(audio, audio_length)
        x_enc, mask = self.encoder(x_enc, mask)  # (batch, frame, encoder_size)
        x_ctc = self.linear(x_enc).log_softmax(-1).transpose(0, 1)  # (frame, batch, vocab_size)
        x_dec, _ = self.predictor(token, self.predictor.init_state(b, x_enc.device))  # (batch, time, predictor_size)
        x_rnnt = self.joiner(x_enc[:, :, None, :], x_dec[:, None, :, :])  # (batch, frame, time, vocab_size)
        #  loss
        frame_length = mask.sum(-1).to(torch.int32)
        target_ctc = target[target != self.blank_token_id]  # (batch * time',)
        rnnt_loss = self.rnnt_loss(x_rnnt, target, frame_length, target_length)
        ctc_loss = self.ctc_loss(x_ctc, target_ctc, frame_length, target_length) / b
        loss = rnnt_loss + self.ctc_loss_weight * ctc_loss
        return loss, {"loss": loss.item(), "rnnt_loss": rnnt_loss.item(), "ctc_loss": ctc_loss.item()}
