import math
import os
from logging import getLogger

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks.tasks.lm.collate_fn import CollateFn
from neural_networks.tasks.lm.dataset import CustomDataset
from neural_networks.tasks.lm.model import Model
from neural_networks.utils.tokenizer import Tokenizer

logger = getLogger(__name__)


@torch.no_grad()
@hydra.main(version_base=None, config_path=f"{os.path.dirname(__file__)}/../../config", config_name="lm")
def main(config: DictConfig):
    os.makedirs(config.decode.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = CustomDataset(config.dataset.test_json_path)
    tokenizer = Tokenizer(config.tokenizer.model_path)
    state_dict = torch.load(config.decode.model_path, map_location=device)
    model = Model(**config.model).to(device)
    model.load_state_dict(state_dict)
    collate_fn = CollateFn(tokenizer, config.model.bos_token_id, config.model.eos_token_id, config.model.pad_token_id)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, **config.dataloader.test)
    total_loss = total_length = 0
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        _, stats = model(**{k: v.to(device) for k, v in batch.items()})
        length = batch["token"].shape[1]
        total_length += length
        total_loss += stats["loss"] * length
    metric = math.exp(total_loss / total_length)
    logger.info(f"ppl: {metric:.5f}")
    with open(f"{config.decode.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"ppl: {metric:.5f}")


if __name__ == "__main__":
    main()
