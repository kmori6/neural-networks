import pytest
import torch

from neural_networks.utils.mask import causal_mask, chunk_mask, sequence_mask


@pytest.mark.parametrize(
    "length, desired",
    [
        (torch.tensor([2, 3]), torch.tensor([[True, True, False], [True, True, True]])),
        (torch.tensor([1, 1, 1]), torch.tensor([[True], [True], [True]])),
        (torch.tensor([1, 3, 2]), torch.tensor([[True, False, False], [True, True, True], [True, True, False]])),
        (torch.tensor([0, 2]), torch.tensor([[False, False], [True, True]])),
    ],
)
def test_sequence_mask(length: torch.Tensor, desired: torch.Tensor):
    actual = sequence_mask(length)
    assert torch.equal(actual, desired), f"Expected {desired} but got {actual}"


@pytest.mark.parametrize(
    "length, chunk_size, history_window_size, desired",
    [
        (
            6,
            2,
            0,
            torch.tensor(
                [
                    [True, True, False, False, False, False],
                    [True, True, False, False, False, False],
                    [False, False, True, True, False, False],
                    [False, False, True, True, False, False],
                    [False, False, False, False, True, True],
                    [False, False, False, False, True, True],
                ]
            ),
        ),
        (
            6,
            2,
            1,
            torch.tensor(
                [
                    [True, True, False, False, False, False],
                    [True, True, False, False, False, False],
                    [False, True, True, True, False, False],
                    [False, True, True, True, False, False],
                    [False, False, False, True, True, True],
                    [False, False, False, True, True, True],
                ]
            ),
        ),
        (
            5,
            2,
            0,
            torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [False, False, True, True, False],
                    [False, False, True, True, False],
                    [False, False, False, False, True],
                ]
            ),
        ),
        (
            5,
            2,
            1,
            torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [False, True, True, True, False],
                    [False, True, True, True, False],
                    [False, False, False, True, True],
                ]
            ),
        ),
    ],
)
def test_chunk_mask(length: int, chunk_size: int, history_window_size: int, desired: torch.Tensor):
    actual = chunk_mask(length, chunk_size, history_window_size)
    assert torch.equal(actual, desired), f"Expected {desired} but got {actual}"


@pytest.mark.parametrize(
    "length, desired",
    [
        (
            torch.tensor([2, 3]),
            torch.tensor(
                [
                    [[True, False, False], [True, True, False], [True, True, False]],
                    [[True, False, False], [True, True, False], [True, True, True]],
                ]
            ),
        ),
        (
            torch.tensor([1, 4]),
            torch.tensor(
                [
                    [
                        [True, False, False, False],
                        [True, False, False, False],
                        [True, False, False, False],
                        [True, False, False, False],
                    ],
                    [
                        [True, False, False, False],
                        [True, True, False, False],
                        [True, True, True, False],
                        [True, True, True, True],
                    ],
                ]
            ),
        ),
        (
            torch.tensor([3, 3]),
            torch.tensor(
                [
                    [[True, False, False], [True, True, False], [True, True, True]],
                    [[True, False, False], [True, True, False], [True, True, True]],
                ]
            ),
        ),
    ],
)
def test_causal_mask(length: torch.Tensor, desired: torch.Tensor):
    actual = causal_mask(length)
    assert torch.equal(actual, desired), f"Expected {desired}, but got {actual}"
