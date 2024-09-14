from pathlib import Path

import pytest

from neural_networks.modules.tokenizer import SentencePieceTokenizer, train_tokenizer


@pytest.fixture
def mock_text_file(tmp_path: Path) -> Path:
    text_file_path = tmp_path / "sample.txt"
    with open(text_file_path, "w") as f:
        f.write("This is a test sentence.\n" * 100)
    return text_file_path


def test_sentencepiece_tokenizer(mock_text_file: Path, tmp_path: Path):
    # train
    train_tokenizer(
        text_file_path=str(mock_text_file),
        vocab_size=16,
        model_type="unigram",
        model_prefix=str(tmp_path / "sp_model"),
        character_coverage=1.0,
        input_sentence_size=1000,
    )
    model_file = tmp_path / "sp_model.model"
    vocab_file = tmp_path / "sp_model.vocab"
    assert model_file.exists()
    assert vocab_file.exists()

    # tokenize
    tokenizer = SentencePieceTokenizer(str(model_file))
    text = "This is a test sentence."
    encoded = tokenizer.encode(text)
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == "This is a test sentence."
