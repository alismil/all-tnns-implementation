import logging
import os
import time

import torch
from config import ModelConfig, TrainConfig
from data_loader import create_dataset
from fvcore.nn import FlopCountAnalysis
from loss_functions import all_tnn_loss
from model import AllTnn
from torch.utils.data import DataLoader, RandomSampler

import wandb

logging.basicConfig(level=logging.INFO)

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
        self.cfg = train_config

        if self.cfg.resume_from_checkpoint:
            checkpoint = torch.load(
                self.cfg.checkpoint_to_resume_from, map_location=self.cfg.device
            )

            checkpoint_model_config = checkpoint["model_config"]
            self.model = AllTnn(checkpoint_model_config, self.cfg.device).to(
                self.cfg.device
            )
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, _ in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
            )

            self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.begin_iter = checkpoint["iter_num"]
            self.begin_epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]

        else:
            self.model_cfg = model_config
            self.model = AllTnn(model_config, self.cfg.device).to(self.cfg.device)
            self.all_alpha = [layer.alpha for layer in model_config.layers]

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
            )

            self.begin_iter = 0
            self.begin_epoch = 0
            self.best_val_loss = 1e9

        os.makedirs(self.cfg.out_dir, exist_ok=True)

        dataset = create_dataset(
            self.cfg.dataset_path,
            input_size=self.cfg.input_size,
            min_scale=self.cfg.min_scale,
            hflip=self.cfg.hflip,
            color_jitter=self.cfg.color_jitter,
            keep_in_memory=self.cfg.keep_in_memory,
        )

        self.loaders = {}
        for split, ds in dataset.items():  # ['train', 'validation']
            self.loaders[split] = DataLoader(
                ds,
                batch_size=(
                    self.cfg.train_batch_size
                    if split == "train"
                    else self.cfg.val_batch_size
                ),
                sampler=RandomSampler(ds) if self.cfg.shuffle else None,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=self.cfg.drop_last,
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

        for split in ["train", "validation"]:
            losses = torch.empty(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                data = next(iter(self.loaders[split]))

                inputs = data["image"].to(self.cfg.device)
                labels = data["label"].to(self.cfg.device)

                with self.cfg.ctx:
                    outputs, all_layer_weights, all_layer_dims = self.model(inputs)

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
        logging.info(f"Params: {param_count / 1e6:.0f} M")

        # TODO FlopCountAnalysis does not work for input of type ParameterList
        # flop_count = self.get_flops(
        #     self.model, self.loaders["validation"], self.cfg.device
        # )
        # logging.info(f"Params:{param_count / 1e6:.0f}M, FLOPs: {flop_count / 1e6:.0f}M")

        t0 = time.time()

        for e in range(self.begin_epoch, self.cfg.num_epochs):
            for i, data in enumerate(self.loaders["train"], self.begin_iter):
                self.model.train()

                inputs = data["image"].to(self.cfg.device)
                labels = data["label"].to(self.cfg.device)

                one = time.time()
                outputs, all_layer_weights, all_layer_dims = self.model(inputs)
                two = time.time()
                print("forward time: ", two - one)
                loss = all_tnn_loss(
                    outputs,
                    labels,
                    all_layer_weights,
                    all_layer_dims,
                    self.all_alpha,
                )
                three = time.time()
                print("loss time: ", three - two)
                loss.backward()
                four = time.time()
                print("loss backward time: ", four - three)
                self.optimizer.step()
                five = time.time()
                print("optimizer step time: ", five - four)
                self.optimizer.zero_grad()

                t1 = time.time()

                if i % self.cfg.log_interval == 0:
                    if self.cfg.wandb_log:
                        wandb.log({"train_loss": loss})
                    logging.info(
                        f"epoch {e}, iter {i} train loss: {loss.item():.4f}, time: {t1-t0:.2f}s"
                    )

                t0 = t1

                if i % self.cfg.eval_interval == 0 and i > 0:
                    losses = self.estimate_loss()
                    logging.info(
                        f"epoch {e}, iter {i}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}"
                    )
                    if self.cfg.wandb_log:
                        wandb.log(
                            {
                                "train_loss": losses["train"],
                                "val_loss": losses["validation"],
                            }
                        )
                    if losses["validation"] < self.best_val_loss:
                        self.best_val_loss = losses["validation"]
                        if i > 0:
                            checkpoint = {
                                "model": self.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "model_config": self.model_cfg,
                                "epoch": e,
                                "iter_num": i,
                                "best_val_loss": self.best_val_loss,
                                "train_config": self.cfg,
                            }
                            logging.info(f"saving checkpoint to {self.cfg.out_dir}")
                            torch.save(
                                checkpoint,
                                os.path.join(
                                    self.cfg.out_dir,
                                    f"epoch_{e}_iter_{i}_val_loss_{losses['validation']:.2f}_{self.cfg.wandb_run_name}.pt",
                                ),
                            )


trainer = Trainer(train_config=TrainConfig(), model_config=ModelConfig())

if __name__ == "__main__":
    trainer.train()
