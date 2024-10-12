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

    assert "enc_token" in batch
    assert "enc_token_length" in batch
    assert "dec_token" in batch
    assert "dec_token_length" in batch
    assert "tgt_token" in batch

    enc_token = batch["enc_token"]
    enc_token_length = batch["enc_token_length"]
    dec_token = batch["dec_token"]
    dec_token_length = batch["dec_token_length"]
    tgt_token = batch["tgt_token"]

    assert enc_token.shape == (2, 9)
    assert enc_token_length.tolist() == [6, 9]
    assert dec_token.shape == (2, 10)  # add bos token
    assert dec_token_length.tolist() == [7, 10]
    assert tgt_token.shape == (2, 10)  # add eos token

    exp_enc_token = [
        [ord("s"), ord("r"), ord("c"), ord(" "), ord("h"), ord("i"), 0, 0, 0],
        [ord("s"), ord("r"), ord("c"), ord(" "), ord("h"), ord("e"), ord("l"), ord("l"), ord("o")],
    ]
    exp_dec_token = [
        [1, ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("i"), 0, 0, 0],
        [1, ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("e"), ord("l"), ord("l"), ord("o")],
    ]
    exp_tgt_token = [
        [ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("i"), 2, -100, -100, -100],
        [ord("t"), ord("g"), ord("t"), ord(" "), ord("h"), ord("e"), ord("l"), ord("l"), ord("o"), 2],
    ]
    assert enc_token.tolist() == exp_enc_token
    assert dec_token.tolist() == exp_dec_token
    assert tgt_token.tolist() == exp_tgt_token
