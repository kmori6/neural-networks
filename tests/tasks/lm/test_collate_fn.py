from neural_networks.tasks.lm.collate_fn import CollateFn


class MockSentencePieceTokenizer:
    def __init__(self):
        pass

    def encode(self, text: str) -> list:
        return [ord(c) for c in text]


def test_collate_fn():
    dataset = [{"text": "hi"}, {"text": "hello"}]
    collate_fn = CollateFn(
        MockSentencePieceTokenizer(), bos_token_id=1, eos_token_id=2, pad_token_id=0, ignore_token_id=-100
    )
    batch = collate_fn(dataset)

    assert "token" in batch
    assert "token_length" in batch
    assert "target" in batch

    token = batch["token"]
    token_length = batch["token_length"]
    target = batch["target"]

    assert token.shape == (2, 6)  # add bos token
    assert token_length.tolist() == [3, 6]
    assert target.shape == (2, 6)

    exp_token = [
        [1, ord("h"), ord("i"), 0, 0, 0],
        [1, ord("h"), ord("e"), ord("l"), ord("l"), ord("o")],
    ]
    exp_target = [
        [ord("h"), ord("i"), 2, -100, -100, -100],
        [ord("h"), ord("e"), ord("l"), ord("l"), ord("o"), 2],
    ]
    assert token.tolist() == exp_token
    assert target.tolist() == exp_target
