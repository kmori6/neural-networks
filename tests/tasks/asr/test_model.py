import torch

from neural_networks.tasks.asr.model import Model


def test_model():
    speech_length = torch.tensor([8000, 16000], dtype=torch.long)
    target_length = torch.tensor([3, 5], dtype=torch.int32)
    speech = torch.randn(2, max(speech_length), dtype=torch.float32)
    token = torch.tensor([[29, 1, 2, 3, 29, 29], [29, 1, 2, 3, 4, 5]], dtype=torch.int32)
    target = torch.tensor([[1, 2, 3, 29, 29], [1, 2, 3, 4, 5]], dtype=torch.int32)
    model = Model(
        vocab_size=30,
        n_mels=80,
        d_model=256,
        num_heads=4,
        kernel_size=31,
        num_blocks=12,
        hidden_size=512,
        num_layers=2,
        dropout_rate=0.1,
        ctc_loss_weight=0.5,
        chunk_size=0,
        num_history_chunks=0,
    )
    loss, stats = model(speech, speech_length, token, target, target_length)
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    for param in model.parameters():
        assert torch.all(torch.isfinite(param.grad))
    assert isinstance(stats, dict)
    assert "loss" in stats
    assert "rnnt_loss" in stats
    assert "ctc_loss" in stats
    assert isinstance(stats["loss"], float)
    assert isinstance(stats["rnnt_loss"], float)
    assert isinstance(stats["ctc_loss"], float)
