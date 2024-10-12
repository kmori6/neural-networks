import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks

from neural_networks.utils.mask import sequence_mask


class Frontend(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512, n_mels: int = 80):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        fbank = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0,
            f_max=8000,
            n_mels=n_mels,
            sample_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("window", torch.hann_window(win_length, dtype=torch.float32), persistent=False)
        self.register_buffer("fbank", fbank)  # (n_freqs, n_mels)

    def forward(self, speech: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            speech (torch.Tensor): Input sequence (batch, sample).
            length (torch.Tensor): Input length (batch,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Output sequence (batch, frame, n_mels).
        """
        x = torch.stft(
            speech,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=speech.dtype, device=speech.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (batch, n_freqs, frame)
        mask = sequence_mask(1 + length // self.hop_length)  # (batch, frame)
        x = torch.view_as_real(x).transpose(1, 2).pow(2).sum(-1)  # (batch, frame, n_freqs)
        x = torch.matmul(x, self.fbank)  # (batch, frame, n_mels)
        x = x.clamp(min=1e-10).log().masked_fill(~mask[:, :, None], 0.0)
        return x, mask
