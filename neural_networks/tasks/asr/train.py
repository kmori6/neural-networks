import os

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from neural_networks.tasks.asr.collate_fn import CollateFn
from neural_networks.tasks.asr.dataset import CustomDataset
from neural_networks.tasks.asr.model import Model
from neural_networks.tasks.txt.tokenizer import SentencePieceTokenizer
from neural_networks.trainer import Trainer
from neural_networks.utils.config import write_train_config


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig):
    os.makedirs(config.trainer.out_dir, exist_ok=True)
    train_dataset = CustomDataset(config.dataset.train_json_path)
    valid_dataset = CustomDataset(config.dataset.valid_json_path)
    tokenizer = SentencePieceTokenizer(config.tokenizer.model_path)
    model = Model(
        vocab_size=config.model.vocab_size,
        n_mels=config.model.n_mels,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        kernel_size=config.model.kernel_size,
        num_blocks=config.model.num_blocks,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout_rate=config.model.dropout_rate,
        ctc_loss_weight=config.model.ctc_loss_weight,
        chunk_size=config.model.chunk_size,
        num_history_chunks=config.model.num_history_chunks,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=config.dataloader.num_workers,
        collate_fn=CollateFn(tokenizer, config.model.vocab_size - 1),
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=config.dataloader.num_workers,
        collate_fn=CollateFn(tokenizer, config.model.vocab_size - 1),
        pin_memory=True,
        drop_last=False,
    )
    write_train_config(config, config.trainer.out_dir)
    trainer = Trainer(model, train_dataloader, valid_dataloader, config.trainer)
    trainer.train()


if __name__ == "__main__":
    main()
