import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.spm_model = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> list:
        return self.spm_model.encode(text)

    def decode(self, ids: list) -> str:
        return self.spm_model.decode(ids)


def train_tokenizer(
    text_file_path: str,
    vocab_size: int,
    model_type: str = "unigram",
    model_prefix: str = "sp",
    character_coverage: float = 1.0,
    input_sentence_size: int = 100000000,
):
    spm.SentencePieceTrainer.train(
        input=text_file_path,
        vocab_size=vocab_size - 1,  # last id is for <blank>
        model_type=model_type,
        model_prefix=model_prefix,
        character_coverage=character_coverage,
        input_sentence_size=input_sentence_size,
        unk_id=0,
        pad_id=-1,  # disable
        bos_id=-1,  # disable
        eos_id=-1,  # disable
    )
