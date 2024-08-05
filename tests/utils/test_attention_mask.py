import pytest
import torch
from neural_networks.utils.attention_mask import sequence_mask


@pytest.mark.parametrize(
    "length, desired",
    [
        (torch.tensor([2, 3]), torch.tensor([[True, True, False], [True, True, True]])),
        (torch.tensor([1, 1, 1]), torch.tensor([[True], [True], [True]])),
        (torch.tensor([1, 3, 2]), torch.tensor([[True, False, False], [True, True, True], [True, True, False]])),
        (torch.tensor([0, 2]), torch.tensor([[False, False], [True, True]])),
    ],
)
def test_make_sequence_mask(length: torch.Tensor, desired: torch.Tensor):
    actual = sequence_mask(length)
    assert torch.equal(actual, desired), f"Expected {desired} but got {actual}"
