import torch

from neural_networks.tasks.mt.model import Model


def test_model():
    enc_token = torch.tensor([[11, 12, 13, 0, 0], [11, 12, 13, 14, 15]], dtype=torch.long)
    enc_token_length = torch.tensor([3, 5], dtype=torch.long)
    dec_token = torch.tensor([[1, 21, 22, 23, 0, 0], [1, 21, 22, 23, 24, 25]], dtype=torch.long)
    dec_token_length = torch.tensor([4, 6], dtype=torch.long)
    tgt_token = torch.tensor([[21, 22, 23, 2, 0, 0], [21, 22, 23, 24, 25, 2]], dtype=torch.long)
    model = Model(
        vocab_size=30,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=3,
        dropout_rate=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        label_smoothing=0.1,
        ignore_token_id=-100,
    )
    loss, stats = model(enc_token, enc_token_length, dec_token, dec_token_length, tgt_token)
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    for param in model.parameters():
        assert torch.all(torch.isfinite(param.grad))
    assert isinstance(stats, dict)
    assert "loss" in stats
    assert "acc" in stats
    assert isinstance(stats["loss"], float)
    assert isinstance(stats["acc"], float)
