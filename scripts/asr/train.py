import os

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from neural_networks.tasks.asr.collate_fn import CollateFn
from neural_networks.tasks.asr.dataset import CustomDataset
from neural_networks.tasks.asr.model import Model
from neural_networks.utils.tokenizer import Tokenizer
from neural_networks.utils.trainer import Trainer


@hydra.main(version_base=None, config_path=f"{os.path.dirname(__file__)}/../../config", config_name="asr")
def main(config: DictConfig):
    os.makedirs(config.trainer.out_dir, exist_ok=True)
    train_dataset = CustomDataset(config.dataset.train_json_path)
    valid_dataset = CustomDataset(config.dataset.valid_json_path)
    tokenizer = Tokenizer(config.tokenizer.model_path)
    model = Model(**config.model)
    collate_fn = CollateFn(tokenizer, config.model.vocab_size - 1)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, **config.dataloader.train)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, **config.dataloader.valid)
    trainer = Trainer(model, train_dataloader, valid_dataloader, config.trainer)
    trainer.train()


if __name__ == "__main__":
    main()