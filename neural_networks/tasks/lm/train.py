import os

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from neural_networks.tasks.lm.collate_fn import CollateFn
from neural_networks.tasks.lm.dataset import CustomDataset
from neural_networks.tasks.lm.model import Model
from neural_networks.tasks.trainer import Trainer
from neural_networks.tasks.txt.tokenizer import SentencePieceTokenizer
from neural_networks.utils.config import write_train_config


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig):
    os.makedirs(config.trainer.out_dir, exist_ok=True)
    train_dataset = CustomDataset(config.dataset.train_json_path)
    valid_dataset = CustomDataset(config.dataset.valid_json_path)
    tokenizer = SentencePieceTokenizer(config.tokenizer.model_path)
    model = Model(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        num_layers=config.model.num_layers,
        dropout_rate=config.model.dropout_rate,
        pad_token_id=config.model.pad_token_id,
        bos_token_id=config.model.bos_token_id,
        eos_token_id=config.model.eos_token_id,
        label_smoothing=config.model.label_smoothing,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=config.dataloader.num_workers,
        collate_fn=CollateFn(
            tokenizer, config.model.bos_token_id, config.model.eos_token_id, config.model.pad_token_id
        ),
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=config.dataloader.num_workers,
        collate_fn=CollateFn(
            tokenizer, config.model.bos_token_id, config.model.eos_token_id, config.model.pad_token_id
        ),
        pin_memory=True,
        drop_last=False,
    )
    write_train_config(config, config.trainer.out_dir)
    trainer = Trainer(model, train_dataloader, valid_dataloader, config.trainer)
    trainer.train()


if __name__ == "__main__":
    main()
