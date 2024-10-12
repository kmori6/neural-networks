import pytest
import torch

from neural_networks.modules.transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    RelativePositionalMultiHeadAttention,
    RotaryPositionalMultiHeadAttention,
)
from neural_networks.utils.mask import causal_mask, sequence_mask


@pytest.fixture
def mock_data() -> dict[str, torch.Tensor]:
    d_model = 64
    time1 = 20
    time2 = 10
    return {
        "x_enc": torch.randn(2, time1, d_model),
        "x_dec": torch.randn(2, time2, d_model),
        "mask_enc": sequence_mask(torch.tensor([time1 // 2, time1]))[:, None, None, :],
        "mask_dec": causal_mask(torch.tensor([time2 // 2, time2]))[:, None, :, :],
    }


def test_positional_encoding(mock_data: dict[str, torch.Tensor]):
    _, seq_length, d_model = mock_data["x_enc"].shape
    module = PositionalEncoding(d_model)
    output = module(mock_data["x_enc"].shape[1], mock_data["x_enc"].dtype, mock_data["x_enc"].device)
    assert output.shape == (seq_length, d_model)


def test_multi_head_attention(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = MultiHeadAttention(d_model, num_heads=8, dropout_rate=0.1)
    # self-attention
    output = module(mock_data["x_enc"], mock_data["x_enc"], mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
    # masked self-attention
    output = module(mock_data["x_dec"], mock_data["x_dec"], mock_data["x_dec"], mock_data["mask_dec"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
    # cross-attention
    output = module(mock_data["x_dec"], mock_data["x_enc"], mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_relative_positional_multi_head_attention(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = RelativePositionalMultiHeadAttention(d_model, num_heads=8, dropout_rate=0.1)
    # self-attention
    output = module(mock_data["x_enc"], mock_data["x_enc"], mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
    # masked self-attention
    output = module(mock_data["x_dec"], mock_data["x_dec"], mock_data["x_dec"], mock_data["mask_dec"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
    # cross-attention
    output = module(mock_data["x_dec"], mock_data["x_enc"], mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_rotary_positional_multi_head_attention(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = RotaryPositionalMultiHeadAttention(d_model, num_heads=8, dropout_rate=0.1)
    # self-attention
    output = module(mock_data["x_enc"], mock_data["x_enc"], mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
    # masked self-attention
    output = module(mock_data["x_dec"], mock_data["x_dec"], mock_data["x_dec"], mock_data["mask_dec"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
    # cross-attention
    output = module(mock_data["x_dec"], mock_data["x_enc"], mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_feed_forward(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = FeedForward(d_model, 4 * d_model, dropout_rate=0.1)
    output = module(mock_data["x_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_encoder_layer(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = EncoderLayer(d_model, num_heads=8, d_ff=4 * d_model, dropout_rate=0.1)
    output = module(mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_decoder_layer(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_dec"].shape
    module = DecoderLayer(d_model, num_heads=8, d_ff=4 * d_model, dropout_rate=0.1)
    output = module(mock_data["x_enc"], mock_data["x_dec"], mock_data["mask_enc"], mock_data["mask_dec"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_encoder(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = Encoder(d_model, num_heads=8, d_ff=4 * d_model, num_layers=3, dropout_rate=0.1)
    output = module(mock_data["x_enc"], mock_data["mask_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_decoder(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_dec"].shape
    module = Decoder(d_model, num_heads=8, d_ff=4 * d_model, num_layers=3, dropout_rate=0.1)
    output = module(mock_data["x_enc"], mock_data["x_dec"], mock_data["mask_enc"], mock_data["mask_dec"])
    assert output.shape == mock_data["x_dec"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
