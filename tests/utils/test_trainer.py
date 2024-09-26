from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from neural_networks.utils.trainer import Trainer


class MockModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        x = self.linear(x)
        return x.sum(), {"loss": 0.5}


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    x_list, y_list = [], []
    for sample in batch:
        x, y = sample
        x_list.append(x)
        y_list.append(y)
    return {"x": torch.stack(x_list, dim=0), "y": torch.stack(y_list, dim=0)}


@pytest.fixture
def trainer() -> Trainer:
    config = OmegaConf.create(
        {
            "out_dir": "",
            "checkpoint_path": "",
            "optimizer": {"lr": 0.001, "weight_decay": 0.01},
            "scheduler": {"warmup_steps": 100},
            "grad_accum_steps": 2,
            "epochs": 3,
            "max_norm": 1.0,
            "log_steps": 1,
        }
    )
    model = MockModel(10, 2)
    dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 2))
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    return Trainer(model, train_dataloader=dataloader, valid_dataloader=dataloader, config=config)


def test_optimizer_creation(trainer: Trainer):
    optimizer = trainer._get_optimizer()
    assert isinstance(optimizer, optim.AdamW)


def test_scheduler_creation(trainer: Trainer):
    scheduler = trainer._get_scheduler()
    assert isinstance(scheduler, optim.lr_scheduler.LambdaLR)


def test_train_epoch(trainer):
    train_loss = trainer._train_epoch(epoch=0)
    assert isinstance(train_loss, float)


def test_validate_epoch(trainer):
    valid_loss = trainer._validate_epoch(epoch=0)
    assert isinstance(valid_loss, float)


def test_checkpoint_save_load(trainer: Trainer, tmp_path: Path):
    trainer.best_valid_loss = 0.1
    trainer.out_dir = tmp_path
    checkpoint_path = tmp_path / "checkpoint_epoch_1.pt"
    best_model_path = tmp_path / "best_model.pt"
    trainer._save_checkpoint(epoch=1, is_best=False)
    assert checkpoint_path.exists()
    assert not best_model_path.exists()

    config = OmegaConf.create(
        {
            "out_dir": "",
            "checkpoint_path": str(checkpoint_path),
            "optimizer": {"lr": 0.001, "weight_decay": 0.01},
            "scheduler": {"warmup_steps": 100},
            "grad_accum_steps": 2,
            "epochs": 3,
            "max_norm": 1.0,
            "log_steps": 1,
        }
    )
    model = MockModel(10, 2)
    dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 2))
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    trainer = Trainer(model, train_dataloader=dataloader, valid_dataloader=dataloader, config=config)
    assert trainer.best_valid_loss == 0.1
    assert trainer.start_epoch == 1


def test_train(trainer: Trainer, tmp_path: Path):
    trainer.out_dir = tmp_path
    trainer.train()
    assert trainer.best_valid_loss <= 0.5
