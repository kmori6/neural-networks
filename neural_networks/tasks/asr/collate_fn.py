from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from neural_networks.tasks.txt.tokenizer import SentencePieceTokenizer


class CollateFn:
    def __init__(self, tokenizer: SentencePieceTokenizer, blank_token_id: int):
        self.tokenizer = tokenizer
        self.blank_token_id = blank_token_id

    def __call__(self, sample_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        speech_list, speech_length_list, token_list, target_list, target_length_list = [], [], [], [], []
        for sample in sample_list:
            speech = sample["speech"]
            speech_list.append(speech)
            speech_length_list.append(len(speech))
            target = self.tokenizer.encode(sample["text"])
            token = [self.blank_token_id] + target
            token_list.append(torch.tensor(token, dtype=torch.long))
            target_list.append(torch.tensor(target, dtype=torch.int32))
            target_length_list.append(len(target))
        batch = {
            "speech": pad_sequence(speech_list, True, 0.0),
            "speech_length": torch.tensor(speech_length_list, dtype=torch.long),
            "token": pad_sequence(token_list, True, self.blank_token_id),
            "target": pad_sequence(target_list, True, self.blank_token_id),
            "target_length": torch.tensor(target_length_list, dtype=torch.int32),
        }
        return batch
