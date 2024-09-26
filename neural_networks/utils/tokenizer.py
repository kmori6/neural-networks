from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp_model = SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> list:
        return self.sp_model.encode(text)

    def decode(self, ids: list) -> str:
        return self.sp_model.decode(ids)


def train_tokenizer(
    text_file_path: str,
    vocab_size: int,
    model_type: str = "unigram",
    model_prefix: str = "sp",
    character_coverage: float = 1.0,
    input_sentence_size: int = 100000000,
):
    SentencePieceTrainer.train(
        input=text_file_path,
        vocab_size=vocab_size - 1,  # last id is for <blank>
        model_type=model_type,
        model_prefix=model_prefix,
        character_coverage=character_coverage,
        input_sentence_size=input_sentence_size,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )
