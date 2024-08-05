import pytest
import torch
from neural_networks.modules.joiner import Joiner


@pytest.fixture
def joiner_model() -> Joiner:
    vocab_size = 30
    encoder_size = 512
    predictor_size = 640
    joiner_size = 640
    return Joiner(vocab_size, encoder_size, predictor_size, joiner_size)


def test_joiner(joiner_model: Joiner):
    batch_size = 4
    frame_length = 20
    seq_length = 10
    encoder_size = 512
    predictor_size = 640
    vocab_size = 30

    encoder_output = torch.randn(batch_size, frame_length, 1, encoder_size)
    predictor_output = torch.randn(batch_size, 1, seq_length, predictor_size)

    # check forward
    output = joiner_model(encoder_output, predictor_output)
    assert output.shape == (batch_size, frame_length, seq_length, vocab_size)

    # check backward
    loss = output.sum()
    loss.backward()
    for param in joiner_model.parameters():
        assert param.grad is not None
