import json
import math
from argparse import Namespace
from logging import getLogger

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks.tasks.lm.collate_fn import CollateFn
from neural_networks.tasks.lm.dataset import CustomDataset
from neural_networks.tasks.lm.model import Model
from neural_networks.tasks.txt.tokenizer import SentencePieceTokenizer

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="evaluate")
def main(config: DictConfig):
    with open(config.train_config_path, "r", encoding="utf-8") as f:
        train_config = Namespace(**json.load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(**train_config.tokenizer)
    test_dataset = CustomDataset(config.test_json_path)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CollateFn(
            tokenizer,
            train_config.model["bos_token_id"],
            train_config.model["eos_token_id"],
            train_config.model["pad_token_id"],
        ),
        drop_last=False,
    )
    state_dict = torch.load(config.model_path, map_location=device)
    model = Model(**train_config.model).to(device).eval()
    model.load_state_dict(state_dict)
    total_loss = total_length = 0
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        _, stats = model(**{k: v.to(device) for k, v in batch.items()})
        length = batch["token"].shape[1]
        total_length += length
        total_loss += stats["loss"] * length
    metric = math.exp(total_loss / total_length)
    logger.info(f"pp: {metric:.5f}")


if __name__ == "__main__":
    main()
