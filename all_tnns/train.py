import logging
import os
import time

import torch
import wandb
from config import ModelConfig, TrainConfig
from datasets import load_dataset
from fvcore.nn import FlopCountAnalysis
from loss_functions import all_tnn_loss
from model import AllTnn
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
        self.cfg = train_config

        self.model_cfg = model_config
        self.model = AllTnn(model_config).to(self.cfg.device)
        self.all_alpha = [layer.alpha for layer in model_config.layers]

        os.makedirs(self.cfg.out_dir, exists_ok=True)

        train_dataset = load_dataset(
            "kietzmannlab/ecoset", streaming=True, split="train"
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=2,
        )

        test_dataset = load_dataset("kietzmannlab/ecoset", streaming=True, split="test")
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=True,
            num_workers=2,
        )

    @torch.no_grad()
    def get_flops(
        self, model: torch.nn.Module, loader: DataLoader, device: torch.device
    ):
        model.eval()
        batch = next(iter(loader))
        x = batch["image"][:1].to(device)
        flops = FlopCountAnalysis(model, x)
        return flops.total()

    @torch.no_grad()
    def estimate_loss(self):
        """Calculates average loss over eval_iters mini batches from test set"""
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                data = next(self.train_dataloader)
                inputs, labels = data

                with self.cfg.ctx:
                    outputs, all_layer_weights, all_layer_dims = self.model(
                        inputs.to(self.cfg.device)
                    )
                    loss = all_tnn_loss(
                        outputs,
                        labels,
                        all_layer_weights,
                        all_layer_dims,
                        self.all_alpha,
                    )

                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        """Main training loop"""

        if self.cfg.wandb_log:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=self.cfg,
            )
            wandb.watch(self.model)

        param_count = sum(p.numel() for p in self.model.parameters())
        flop_count = self.get_flops(self.model, self.test_dataloader, self.cfg.device)
        logging.info("Params: %.0fM, FLOPs: %.0fM", param_count / 1e6, flop_count / 1e6)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            eps=self.cfg.eps,
            weight_decay=self.cfg.weight_decay,
        )

        t0 = time.time()
        best_val_loss = 1e9

        for _ in range(self.cfg.num_epochs):
            for i, data in enumerate(self.train_dataloader, 0):
                self.model.train()

                inputs, labels = data

                outputs, all_layer_weights, all_layer_dims = self.model(
                    inputs.to(self.cfg.device)
                )

                loss = all_tnn_loss(
                    outputs,
                    labels,
                    all_layer_weights,
                    all_layer_dims,
                    self.all_alpha,
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                t1 = time.time()

                if i % self.cfg.log_interval == 0:
                    if self.cfg.wandb_log:
                        wandb.log({"train_loss": loss})
                    logging.info(
                        f"iter {i} train loss: {loss.item():.4f}, time: {t1-t0:.2f}s"
                    )

                t0 = t1

                if i % self.cfg.eval_interval == 0:
                    losses = self.estimate_loss()
                    logging.info(
                        f"iter {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )
                    if self.cfg.wandb_log:
                        wandb.log(
                            {
                                "train_loss": losses["train"],
                                "val_loss": losses["val"],
                            }
                        )
                    if losses["val"] < best_val_loss:
                        best_val_loss = losses["val"]
                        if i > 0:
                            checkpoint = {
                                "model": self.model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "model_config": self.model_cfg,
                                "iter_num": i,
                                "best_val_loss": best_val_loss,
                                "train_config": self.cfg,
                            }
                            logging.info(f"saving checkpoint to {self.cfg.out_dir}")
                            torch.save(
                                checkpoint,
                                os.path.join(
                                    self.cfg.out_dir,
                                    f"{i}_{losses['val']:.2f}_{self.cfg.wandb_run_name}.pt",
                                ),
                            )
