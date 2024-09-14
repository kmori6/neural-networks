from typing import Any

import pytest
import torch

from neural_networks.modules.transducer import Joiner, Predictor, default_beam_search


@pytest.fixture
def mock_data() -> dict[str, Any]:
    d_enc = 64
    d_pred = 80
    time1 = 20
    vocab_size = time2 = 30
    return {
        "token": torch.arange(vocab_size, dtype=torch.long).repeat(2).view(2, vocab_size),
        "x_enc": torch.randn(2, time1, 1, d_enc),
        "x_pred": torch.randn(2, 1, time2, d_pred),
    }


def test_predictor(mock_data: dict[str, Any]):
    b, _, time, d_pred = mock_data["x_pred"].shape
    num_layers = 1
    module = Predictor(time, d_pred, num_layers=num_layers, dropout_rate=0.1, blank_token_id=time - 1)
    output1 = module.init_state(b, device=torch.device("cpu"))
    assert isinstance(output1, tuple) and len(output1) == 2
    assert output1[0].shape == (num_layers, b, d_pred)
    assert output1[1].shape == (num_layers, b, d_pred)
    output2, output3 = module(mock_data["token"], output1)
    assert output2.shape == (b, time, d_pred)
    assert isinstance(output3, tuple) and len(output3) == 2
    assert output3[0].shape == (num_layers, b, d_pred)
    assert output3[1].shape == (num_layers, b, d_pred)
    loss = output2.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_joiner(mock_data: dict[str, Any]):
    b, time1, _, d_enc = mock_data["x_enc"].shape
    b, _, time2, d_pred = mock_data["x_pred"].shape
    vocab_size = 30
    module = Joiner(vocab_size, d_enc, d_pred, d_pred)
    output = module(mock_data["x_enc"], mock_data["x_pred"])
    assert output.shape == (b, time1, time2, vocab_size)
    loss = output.sum()
    loss.backward()
    for param in module.parameters():
        assert param.grad is not None and torch.all(torch.isfinite(param.grad))


def test_default_beam_search(mock_data):
    *_, d_enc = mock_data["x_enc"].shape
    *_, d_pred = mock_data["x_pred"].shape
    vocab_size = 30
    predictor = Predictor(vocab_size, d_pred, num_layers=1, dropout_rate=0.1, blank_token_id=vocab_size - 1)
    joiner = Joiner(vocab_size, d_enc, d_pred, d_pred)
    x_enc = torch.randn(5, d_enc)  # (time, d_model)
    beam_size = 3
    B = [{"score": 0.0, "token": [vocab_size - 1], "state": None}]
    cache = {}
    hyps, cache = default_beam_search(x_enc, beam_size, B, cache, predictor, joiner)
    assert len(hyps) >= beam_size
    assert all(isinstance(hyp, dict) for hyp in hyps)
    assert "score" in hyps[0]
    assert "token" in hyps[0]
    assert isinstance(cache, dict)
    assert len(cache) > 0
