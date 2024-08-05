import pytest
import torch
from neural_networks.modules.conformer import Conformer
from neural_networks.utils.attention_mask import sequence_mask


@pytest.fixture
def conformer_model() -> Conformer:
    input_size = 80
    d_model = 144
    num_heads = 4
    kernel_size = 31
    num_blocks = 16
    return Conformer(input_size, d_model, num_heads, kernel_size, num_blocks)


def test_conformer_forward(conformer_model: Conformer):
    batch_size = 2
    seq_length = 100
    input_dim = 80

    speech = torch.randn(batch_size, seq_length, input_dim)
    mask = sequence_mask(torch.tensor([seq_length // 2, seq_length]))[:, None, :]

    # check forward
    output, mask = conformer_model(speech, mask)
    assert len(output.shape) == 3
    assert output.shape[0] == batch_size
    assert output.shape[2] == conformer_model.d_model

    # check backward
    loss = output.sum()
    loss.backward()
    for param in conformer_model.parameters():
        assert param.grad is not None
