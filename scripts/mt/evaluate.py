import os
from logging import getLogger

import hydra
import torch
from omegaconf import DictConfig
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer

from neural_networks.tasks.mt.dataset import CustomDataset
from neural_networks.tasks.mt.model import Model
from neural_networks.utils.tokenizer import Tokenizer

logger = getLogger(__name__)


@torch.no_grad()
def translate(
    model: Model, src_text: str, tokenizer: Tokenizer, beam_size: int, device: torch.device, max_length: int = 1024
) -> str:
    token_enc = torch.tensor([tokenizer.encode(src_text)], dtype=torch.long, device=device)  # (1, time1)
    # encode
    mask_enc = torch.ones_like(token_enc, dtype=torch.bool)[:, None, None, :]  # (1, 1, 1, time1)
    x_enc = model.input_embedding(token_enc)  # (1, time1, d_model)
    x_enc = x_enc * model.scale + model.positional_encoding(x_enc.shape[1], x_enc.dtype, x_enc.device)
    x_enc = model.encoder(x_enc, mask_enc)
    # decode
    hyps = [{"score": 0.0, "token": [model.bos_token_id]}]
    for i in range(max_length):
        best_hyps = []
        for hyp in hyps:
            token_dec = torch.tensor([hyp["token"]], dtype=torch.long, device=device)  # (1, time2)
            mask_dec = torch.ones(
                1, 1, len(hyp["token"]), len(hyp["token"]), dtype=torch.bool, device=device  # type: ignore[arg-type]
            )  # (1, 1, time2, time2)
            x_dec = model.output_embedding(token_dec)  # (batch, time2, d_model)
            x_dec = x_dec * model.scale + model.positional_encoding(x_dec.shape[1], x_dec.dtype, x_dec.device)
            x_dec = model.decoder(x_enc, x_dec, mask_enc, mask_dec)
            x_dec = model.linear(x_dec)  # (1, time2, vocab_size)
            logp = x_dec.log_softmax(-1)[0, -1, :]  # (vocab_size,)
            topk = logp.topk(beam_size)
            for score, k in zip(*topk):
                best_hyps.append(
                    {"score": hyp["score"] + score.item(), "token": hyp["token"] + [int(k)]}  # type: ignore[operator]
                )
            best_hyps = sorted(best_hyps, key=lambda x: x["score"] / len(x["token"]), reverse=True)[:beam_size]
        if best_hyps[0]["token"][-1] == model.eos_token_id:
            break
        hyps = best_hyps
    return tokenizer.decode(best_hyps[0]["token"][1:-1])  # remove bos and eos token ids


@hydra.main(version_base=None, config_path=f"{os.path.dirname(__file__)}/../../config", config_name="mt")
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = CustomDataset(config.dataset.test_json_path)
    tokenizer = Tokenizer(config.tokenizer.model_path)
    state_dict = torch.load(config.decode.model_path, map_location=device)
    model = Model(**config.model).to(device).eval()
    model.load_state_dict(state_dict)
    normalizer = BasicTextNormalizer()
    hyp_list, ref_list = [], []
    with open(f"{config.decode.out_dir}/ref.txt", "w", encoding="utf-8") as f_ref, open(
        f"{config.decode.out_dir}/hyp.txt", "w", encoding="utf-8"
    ) as f_hyp:
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            ref_list.append(normalizer(sample["tgt_text"]))
            f_ref.write(normalizer(sample["tgt_text"]) + "\n")
            hyp = translate(model, sample["src_text"], tokenizer, config.decode.beam_size, device)
            hyp_list.append(normalizer(hyp))
            f_hyp.write(normalizer(hyp) + "\n")
    bleu = BLEU()
    metric = bleu.corpus_score(hyp_list, [ref_list]).score
    logger.info(f"bleu: {metric:.5f}")
    with open(f"{config.decode.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
