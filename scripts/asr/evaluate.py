import os
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
from neural_networks.utils.tokenizer import Tokenizer

logger = getLogger(__name__)


@torch.no_grad()
def recognize(
    model: Model,
    audio: torch.Tensor,
    tokenizer: Tokenizer,
    beam_size: int,
    audio_chunk_size: int,
    history_chunk_size: int,
    history_window_size: int,
) -> str:
    audio_chunks = audio.split(audio_chunk_size)
    history_chunk = torch.zeros(history_chunk_size, dtype=audio.dtype, device=audio.device)
    hyp = [{"score": 0.0, "token": [model.blank_token_id], "state": model.predictor.init_state(1, audio.device)}]
    cache: dict[tuple, Any] = {}
    for audio_chunk in audio_chunks:
        if len(audio_chunk) < audio_chunk_size:
            audio_chunk = torch.cat([audio_chunk, audio.new_zeros(audio_chunk_size - len(audio_chunk))])
        input_chunk = torch.cat([history_chunk, audio_chunk])
        input_length = torch.tensor([len(input_chunk)], dtype=torch.long, device=audio.device)
        x, mask = model.frontend(input_chunk[None, :], input_length)
        x, _ = model.encoder(x, mask)  # (batch, frame, encoder_size)
        hyp, cache = default_beam_search(
            x[0, history_window_size:], beam_size, hyp, cache, model.predictor, model.joiner
        )
        history_chunk = torch.cat([history_chunk[len(audio_chunk) :], audio_chunk])
    return tokenizer.decode(hyp[0]["token"][1:])  # type: ignore[index]


@hydra.main(version_base=None, config_path=f"{os.path.dirname(__file__)}/../../config", config_name="asr")
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = CustomDataset(config.dataset.test_json_path)
    tokenizer = Tokenizer(config.tokenizer.model_path)
    state_dict = torch.load(config.decode.model_path, map_location=device)
    model = Model(**config.model).to(device).eval()
    model.load_state_dict(state_dict)
    model.encoder.chunk_size = -1
    hyp_list, ref_list = [], []
    with open(f"{config.decode.out_dir}/ref.txt", "w", encoding="utf-8") as f_ref, open(
        f"{config.decode.out_dir}/hyp.txt", "w", encoding="utf-8"
    ) as f_hyp:
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            ref_list.append(sample["text"])
            f_ref.write(sample["text"] + "\n")
            hyp = recognize(
                model,
                sample["audio"].to(device),
                tokenizer,
                config.decode.beam_size,
                config.decode.audio_chunk_size,
                config.decode.history_chunk_size,
                config.decode.history_window_size,
            )
            hyp_list.append(hyp)
            f_hyp.write(hyp + "\n")
    normalizer = EnglishTextNormalizer()
    error = total = 0
    for ref, hyp in zip(ref_list, hyp_list):
        output = process_words(normalizer(ref), normalizer(hyp))
        error += output.substitutions + output.deletions + output.insertions
        total += output.substitutions + output.deletions + output.hits
    metric = error / total
    logger.info(f"wer: {metric:.5f}")
    with open(f"{config.decode.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
