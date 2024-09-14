import pytest
import torch

from neural_networks.tasks.asr.collate_fn import CollateFn


# Mock SentencePieceTokenizer
class MockSentencePieceTokenizer:
    def __init__(self, model_path: str):
        pass

    def encode(self, text: str) -> list:
        return [ord(c) for c in text]


@pytest.fixture
def tokenizer():
    return MockSentencePieceTokenizer(model_path="dummy.model")


@pytest.fixture
def collate_fn(tokenizer):
    return CollateFn(tokenizer, blank_token_id=0)


def test_collate_fn(collate_fn):
    dataset = [
        {"speech": torch.tensor([0.1, 0.2, 0.3]), "text": "hi"},
        {"speech": torch.tensor([0.4, 0.5]), "text": "hello"},
    ]

    batch = collate_fn(dataset)

    assert "speech" in batch
    assert "speech_length" in batch
    assert "token" in batch
    assert "target" in batch
    assert "target_length" in batch

    speech = batch["speech"]
    speech_length = batch["speech_length"]
    token = batch["token"]
    target = batch["target"]
    target_length = batch["target_length"]

    assert speech.shape == (2, 3)
    assert speech_length.tolist() == [3, 2]

    expected_target = [[ord("h"), ord("i"), 0, 0, 0], [ord("h"), ord("e"), ord("l"), ord("l"), ord("o")]]
    expected_token = [[0, ord("h"), ord("i"), 0, 0, 0], [0, ord("h"), ord("e"), ord("l"), ord("l"), ord("o")]]

    assert target.shape == (2, 5)
    assert token.shape == (2, 6)
    assert token.tolist() == expected_token
    assert target.tolist() == expected_target
    assert target_length.tolist() == [2, 5]
