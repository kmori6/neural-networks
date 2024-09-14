import pytest
import torch

from neural_networks.modules.conformer import (
    Conformer,
    ConformerBlock,
    ConvolutionModule,
    ConvolutionSubsampling,
    FeedForwardModule,
    MultiHeadSelfAttentionModule,
)
from neural_networks.utils.attention_mask import sequence_mask


@pytest.fixture
def mock_data() -> dict[str, torch.Tensor]:
    n_mels = 10
    d_model = 64
    time = 100
    seq_len = torch.tensor([100 // 2, 100])
    return {
        "x_feat": torch.randn(2, time, n_mels),
        "seq_len": seq_len,
        "x_enc": torch.randn(2, time, d_model),
        "mask_feat": sequence_mask(seq_len),
        "mask_enc": sequence_mask(seq_len)[:, None, None, :],
    }


def test_convolution_subsampling(mock_data: dict[str, torch.Tensor]):
    b, frame, n_mels = mock_data["x_feat"].shape
    *_, d_model = mock_data["x_enc"].shape
    module = ConvolutionSubsampling(d_model)
    output = module(mock_data["x_feat"], mock_data["mask_feat"])
    out_frame = ((frame - 1) // 2 - 1) // 2
    out_channel = d_model * (((n_mels - 1) // 2 - 1) // 2)
    assert isinstance(output, tuple) and len(output) == 2
    assert output[0].shape == (b, out_frame, out_channel)
    assert output[1].shape == (b, out_frame)
    assert torch.equal(output[1].sum(-1), ((mock_data["seq_len"] - 1) // 2 - 1) // 2)
    loss = output[0].sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_feed_forward_module(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = FeedForwardModule(d_model, dropout_rate=0.1)
    output = module(mock_data["x_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_multi_head_self_attention_module(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = MultiHeadSelfAttentionModule(d_model, num_heads=8, dropout_rate=0.1)
    p = torch.randn(1, 2 * mock_data["x_enc"].shape[1] - 1, d_model)
    output = module(mock_data["x_enc"], p, mock_data["mask_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_convolution_module(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = ConvolutionModule(d_model, kernel_size=15, dropout_rate=0.1)
    output = module(mock_data["x_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_conformer_block(mock_data: dict[str, torch.Tensor]):
    *_, d_model = mock_data["x_enc"].shape
    module = ConformerBlock(d_model, num_heads=8, kernel_size=15, dropout_rate=0.1)
    p = torch.randn(1, 2 * mock_data["x_enc"].shape[1] - 1, d_model)
    output = module(mock_data["x_enc"], p, mock_data["mask_enc"])
    assert output.shape == mock_data["x_enc"].shape
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_conformer(mock_data: dict[str, torch.Tensor]):
    b, frame, n_mels = mock_data["x_feat"].shape
    *_, d_model = mock_data["x_enc"].shape
    module = Conformer(n_mels, d_model, num_heads=8, kernel_size=15, num_blocks=3, dropout_rate=0.1)
    output = module(mock_data["x_feat"], mock_data["mask_feat"])
    out_frame = ((frame - 1) // 2 - 1) // 2
    out_channel = d_model * (((n_mels - 1) // 2 - 1) // 2)
    assert isinstance(output, tuple) and len(output) == 2
    assert output[0].shape == (b, out_frame, out_channel)
    assert output[1].shape == (b, out_frame)
    assert torch.equal(output[1].sum(-1), ((mock_data["seq_len"] - 1) // 2 - 1) // 2)
    loss = output[0].sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
    # streaming
    module = Conformer(
        n_mels, d_model, num_heads=8, kernel_size=15, num_blocks=3, dropout_rate=0.1, chunk_size=2, num_history_chunks=1
    )
    output = module(mock_data["x_feat"], mock_data["mask_feat"])
    out_frame = ((frame - 1) // 2 - 1) // 2
    out_channel = d_model * (((n_mels - 1) // 2 - 1) // 2)
    assert isinstance(output, tuple) and len(output) == 2
    assert output[0].shape == (b, out_frame, out_channel)
    assert output[1].shape == (b, out_frame)
    assert torch.equal(output[1].sum(-1), ((mock_data["seq_len"] - 1) // 2 - 1) // 2)
    loss = output[0].sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))
