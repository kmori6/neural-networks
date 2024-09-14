from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from neural_networks.tasks.txt.tokenizer import SentencePieceTokenizer


class CollateFn:
    def __init__(
        self,
        tokenizer: SentencePieceTokenizer,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        ignore_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id

    def __call__(self, sample_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        token_enc_list, enc_length_list, token_dec_list, dec_length_list, token_tgt_list = [], [], [], [], []
        for sample in sample_list:
            token_enc = self.tokenizer.encode(sample["src_text"])
            token_enc_list.append(torch.tensor(token_enc, dtype=torch.long))
            enc_length_list.append(len(token_enc))
            base_token = self.tokenizer.encode(sample["tgt_text"])
            token_dec = [self.bos_token_id] + base_token
            token_tgt = base_token + [self.eos_token_id]
            token_dec_list.append(torch.tensor(token_dec, dtype=torch.long))
            token_tgt_list.append(torch.tensor(token_tgt, dtype=torch.long))
            dec_length_list.append(len(token_dec))
        batch = {
            "token_enc": pad_sequence(token_enc_list, True, self.pad_token_id),
            "token_enc_length": torch.tensor(enc_length_list, dtype=torch.long),
            "token_dec": pad_sequence(token_dec_list, True, self.pad_token_id),
            "token_dec_length": torch.tensor(dec_length_list, dtype=torch.long),
            "token_tgt": pad_sequence(token_tgt_list, True, self.ignore_token_id),
        }
        return batch
