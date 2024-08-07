import os
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

logger = getLogger(__name__)


class Trainer:
    def __init__(
        self, model: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, config: DictConfig
    ):
        assert torch.cuda.is_available()
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = DDP(model.to(self.gpu_id), device_ids=[self.gpu_id])
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = GradScaler()
        self.best_valid_loss = float("inf")
        self.start_epoch = 0

        if os.path.exists(self.config.checkpoint_path):
            self._load_checkpoint()

    def _get_optimizer(self):
        return optim.AdamW(
            self.model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay
        )

    def _get_scheduler(self):
        total_step = len(self.train_dataloader) * self.config.epochs

        def lr_lambda(step: int):
            if step < self.config.scheduler.warmup_steps:
                return step / self.config.scheduler.warmup_steps
            else:
                return (total_step - step) / (total_step - self.config.scheduler.warmup_steps)

        return LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)

    def _train_epoch(self, epoch: int):
        logger.info(f"epoch {epoch + 1} training")
        self.model.train()
        self.optimizer.zero_grad()
        epoch_loss = 0.0
        for step, batch in enumerate(self.train_dataloader, 1):
            with autocast(dtype=torch.bfloat16):
                loss, stats = self.model(**{k: v.to(self.gpu_id) for k, v in batch.items()})
                loss = loss / self.config.grad_accum_steps
            self.scaler.scale(loss).backward()
            if step % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if torch.isfinite(total_norm):
                    self.scheduler.step()
                self.optimizer.zero_grad()
            if self.gpu_id == 0 and step % self.config.log_steps == 0:
                msg = (
                    f"epoch: {epoch + 1}/{self.config.epochs}, "
                    f"step: {step:,}/{len(self.train_dataloader):,}, "
                    f"lr: {self.scheduler.get_last_lr()[0]:.6f}, "
                )
                for k, v in stats.items():
                    msg += f"{k}: {v:.3f}, "
                logger.info(msg)
                break
            epoch_loss += stats["loss"] * batch[next(iter(batch))].shape[0]

        self.scheduler.step()

        return epoch_loss / len(self.train_dataloader)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int):
        logger.info(f"epoch {epoch + 1} validation")
        self.model.eval()
        epoch_loss = 0.0
        for batch in self.valid_dataloader:
            loss, _ = self.model(**{k: v.to(self.gpu_id) for k, v in batch.items()})
            epoch_loss += loss.item() * batch[next(iter(batch))].shape[0]
        return epoch_loss / len(self.valid_dataloader)

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_valid_loss": self.best_valid_loss,
        }
        torch.save(state, Path(self.config.out_dir) / f"checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(self.model.state_dict(), Path(self.config.out_dir) / "best_model.pt")

    def _load_checkpoint(self):
        checkpoint = torch.load(self.config.checkpoint_path, map_location=f"cuda:{self.gpu_id}")
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_valid_loss = checkpoint["best_valid_loss"]
        self.start_epoch = checkpoint["epoch"]

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._train_epoch(epoch)
            if self.gpu_id == 0:
                valid_loss = self._validate_epoch(epoch)
                logger.info(
                    f"epoch {epoch + 1}/{self.config.epochs}, "
                    f"train loss: {train_loss:.3f}, "
                    f"valid loss: {valid_loss:.3f}"
                )
                is_best = valid_loss < self.best_valid_loss
                if is_best:
                    self.best_valid_loss = valid_loss
                self._save_checkpoint(epoch + 1, is_best)
