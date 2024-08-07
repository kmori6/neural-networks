import hydra
from neural_networks.tasks.asr.collate_fn import CollateFn
from neural_networks.tasks.asr.dataset import CustomDataset
from neural_networks.tasks.asr.model import Model
from neural_networks.tasks.asr.sampler import LengthBucketSampler
from neural_networks.tasks.token.sentencepiece_tokenizer import SentencePieceTokenizer
from neural_networks.tasks.trainer import Trainer
from neural_networks.utils.ddp import ddp_setup
from omegaconf import DictConfig
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig):
    ddp_setup()
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
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_dataset),
        num_workers=config.dataloader.num_workers,
        collate_fn=CollateFn(tokenizer, config.model.vocab_size - 1),
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.dataloader.batch_size,
        sampler=LengthBucketSampler(len(valid_dataset), config.dataloader.batch_size, shuffle=False),
        num_workers=config.dataloader.num_workers,
        collate_fn=CollateFn(tokenizer, config.model.vocab_size - 1),
        pin_memory=True,
        drop_last=False,
    )
    trainer = Trainer(model, train_dataloader, valid_dataloader, config.trainer)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
