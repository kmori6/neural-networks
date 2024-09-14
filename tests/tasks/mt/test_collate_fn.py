from neural_networks.tasks.mt.collate_fn import CollateFn


class MockSentencePieceTokenizer:
    def __init__(self):
        pass

    def encode(self, text: str) -> list:
        return [ord(c) for c in text]


def test_collate_fn():
    dataset = [
        {"src_text": "src hi", "tgt_text": "tgt hi"},
        {"src_text": "src hello", "tgt_text": "tgt hello"},
    ]
    collate_fn = CollateFn(
        MockSentencePieceTokenizer(), bos_token_id=1, eos_token_id=2, pad_token_id=0, ignore_token_id=-100
    )
    batch = collate_fn(dataset)

    assert "token_enc" in batch
    assert "token_enc_length" in batch
    assert "token_dec" in batch
    assert "token_dec_length" in batch
    assert "token_tgt" in batch

    token_enc = batch["token_enc"]
    token_enc_length = batch["token_enc_length"]
    token_dec = batch["token_dec"]
    token_dec_length = batch["token_dec_length"]
    token_tgt = batch["token_tgt"]

    assert token_enc.shape == (2, 9)
    assert token_enc_length.tolist() == [6, 9]
    assert token_dec.shape == (2, 10)  # add bos token
    assert token_dec_length.tolist() == [7, 10]
    assert token_tgt.shape == (2, 10)  # add eos token

    exp_token_enc = [
        [ord("s"), ord("r"), ord("c"), ord(" "), ord("h"), ord("i"), 0, 0, 0],
        [ord("s"), ord("r"), ord("c"), ord(" "), ord("h"), ord("e"), ord("l"), ord("l"), ord("o")],
    ]
    exp_token_dec = [
        [1, ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("i"), 0, 0, 0],
        [1, ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("e"), ord("l"), ord("l"), ord("o")],
    ]
    exp_token_tgt = [
        [ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("i"), 2, -100, -100, -100],
        [ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("e"), ord("l"), ord("l"), ord("o"), 2],
    ]
    assert token_enc.tolist() == exp_token_enc
    assert token_dec.tolist() == exp_token_dec
    assert token_tgt.tolist() == exp_token_tgt
