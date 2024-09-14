import json
from argparse import Namespace
from logging import getLogger
from typing import Any

import hydra
import torch
from jiwer import process_words
from omegaconf import DictConfig
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from neural_networks.modules.transducer import default_beam_search
from neural_networks.tasks.asr.dataset import CustomDataset
from neural_networks.tasks.asr.model import Model
from neural_networks.tasks.txt.tokenizer import SentencePieceTokenizer

logger = getLogger(__name__)


@torch.no_grad()
def recognize(model: Model, speech: torch.Tensor, tokenizer: SentencePieceTokenizer, beam_size: int) -> str:
    speech_length = torch.tensor([len(speech)], dtype=torch.long, device=speech.device)
    x, mask = model.frontend(speech[None, :], speech_length)
    x, _ = model.encoder(x, mask)  # (batch, frame, encoder_size)
    hyp = [{"score": 0.0, "token": [model.blank_token_id], "state": model.predictor.init_state(1, x.device)}]
    cache: dict[tuple, Any] = {}
    hyp, _ = default_beam_search(x[0], beam_size, hyp, cache, model.predictor, model.joiner)
    return tokenizer.decode(hyp[0]["token"][1:])  # type: ignore[index]


@torch.no_grad()
def streaming_recognize(
    model: Model,
    speech: torch.Tensor,
    tokenizer: SentencePieceTokenizer,
    beam_size: int,
    speech_chunk_size: int,
    history_chunk_size: int,
) -> str:
    speech_chunks = speech.split(speech_chunk_size)
    history_chunk = torch.zeros(history_chunk_size, dtype=speech.dtype, device=speech.device)
    hyp = [{"score": 0.0, "token": [model.blank_token_id], "state": model.predictor.init_state(1, speech.device)}]
    cache: dict[tuple, Any] = {}
    for speech_chunk in speech_chunks:
        if len(speech_chunk) < speech_chunk_size:
            speech_chunk = torch.cat([speech_chunk, speech.new_zeros(speech_chunk_size - len(speech_chunk))])
        input_chunk = torch.cat([history_chunk, speech_chunk])
        input_length = torch.tensor([len(input_chunk)], dtype=torch.long, device=speech.device)
        x, mask = model.frontend(input_chunk[None, :], input_length)
        x, _ = model.encoder(x, mask)  # (batch, frame, encoder_size)
        hyp, cache = default_beam_search(x[0], beam_size, hyp, cache, model.predictor, model.joiner)
        history_chunk = speech_chunk
    return tokenizer.decode(hyp[0]["token"][1:])  # type: ignore[index]


@hydra.main(version_base=None, config_path="config", config_name="evaluate")
def main(config: DictConfig):
    with open(config.train_config_path, "r", encoding="utf-8") as f:
        train_config = Namespace(**json.load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = CustomDataset(config.test_json_path)
    tokenizer = SentencePieceTokenizer(**train_config.tokenizer)
    state_dict = torch.load(config.model_path, map_location=device)
    model = Model(**train_config.model).to(device).eval()
    model.load_state_dict(state_dict)
    hyp_list, ref_list = [], []
    for i in tqdm(range(len(test_dataset))):
        sample = test_dataset[i]
        ref_list.append(sample["text"])
        if config.streaming:
            hyp = streaming_recognize(
                model,
                sample["speech"].to(device),
                tokenizer,
                config.beam_size,
                config.speech_chunk_size,
                config.history_chunk_size,
            )
        else:
            hyp = recognize(model, sample["speech"].to(device), tokenizer, config.beam_size)
        hyp_list.append(hyp)

    normalizer = EnglishTextNormalizer()
    error = total = 0
    for ref, hyp in zip(ref_list, hyp_list):
        output = process_words(normalizer(ref), normalizer(hyp))
        error += output.substitutions + output.deletions + output.insertions
        total += output.substitutions + output.deletions + output.hits
    metric = error / total
    logger.info(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
