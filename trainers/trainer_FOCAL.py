import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from models.FOCAL import adapt_focal_checkpoint


class Trainer(object):
    """Trainer for FOCAL self-supervised pretraining."""

    def __init__(self, cfg, model, optimizer, criterion, preprocess, save_path, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.preprocess = preprocess
        self.save_path = save_path
        self.device = device
        self.use_amp = bool(getattr(cfg, "focal_amp", True)) and str(device).startswith("cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

    def _build_two_views(self, batch_subseq):
        """
        Args:
            batch_subseq: [B, S, T, C]
        Returns:
            aug1, aug2: [B*S, T, C]
        """
        x = batch_subseq.detach().cpu().numpy().astype(np.float32)

        # Original FOCAL random mode: one random augmenter per batch.
        aug1, _ = self.preprocess.augment_batch(x)
        aug2, _ = self.preprocess.augment_batch(x)

        aug1 = torch.from_numpy(aug1).float().to(self.device)
        aug2 = torch.from_numpy(aug2).float().to(self.device)

        b, s, t, c = aug1.shape
        aug1 = aug1.reshape(b * s, t, c)
        aug2 = aug2.reshape(b * s, t, c)
        return aug1, aug2

    def pretrain(self, data_loader_train, data_loader_vali, model_file=None, data_parallel=False):
        self.load(model_file)

        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        best_loss = 1e6
        model_best = copy.deepcopy(model.state_dict())
        global_step = 0
        eval_interval = max(1, int(getattr(self.cfg, "focal_eval_interval", 10)))

        for e in range(self.cfg.n_epochs):
            model.train()
            loss_sum = 0.0
            time_sum = 0.0

            for batch in data_loader_train:
                batch = batch.to(self.device).float()  # [B, S, T, C]
                start_time = time.time()
                self.optimizer.zero_grad()

                aug1, aug2 = self._build_two_views(batch)
                try:
                    with autocast(enabled=self.use_amp):
                        mod_features1, mod_features2 = model(aug1, aug2, proj_head=True)
                        loss = self.criterion(mod_features1, mod_features2).mean()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except torch.cuda.OutOfMemoryError as exc:
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "FOCAL pretraining ran out of CUDA memory. "
                        "Current effective encoder batch is batch_size x focal_seq_len x 2 views"
                        f"{' x 2 modalities' if getattr(self.model.backbone, 'use_dual_modalities', True) else ''}. "
                        f"With the current config this expands to {batch.shape[0]} x "
                        f"{getattr(self.cfg, 'focal_seq_len', 1)} windows before the FOCAL loss. "
                        "Try lowering `batch_size` in `config/FOCAL.json` to 32 or 16, "
                        "or temporarily disable `focal_use_dual_modalities` if you need a quick sanity run."
                    ) from exc

                time_sum += time.time() - start_time
                loss_sum += loss.item()
                global_step += 1

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    self.model.load_state_dict(model_best)
                    return

            # Keep FOCAL-style sparse validation cadence.
            if (e + 1) % eval_interval == 0 or (e + 1) == self.cfg.n_epochs:
                loss_eva = self.evaluate(data_loader_vali, data_parallel=data_parallel)
                print(
                    "Epoch %d/%d : Average Loss %5.4f. Vali Loss %5.4f"
                    % (e + 1, self.cfg.n_epochs, loss_sum / max(len(data_loader_train), 1), loss_eva)
                )

                if loss_eva < best_loss:
                    best_loss = loss_eva
                    model_best = copy.deepcopy(model.state_dict())
                    self.save(0)
            else:
                print(
                    "Epoch %d/%d : Average Loss %5.4f."
                    % (e + 1, self.cfg.n_epochs, loss_sum / max(len(data_loader_train), 1))
                )

        self.model.load_state_dict(model_best)
        print("The Total Epoch have been reached.")

    def evaluate(self, data_loader, model_file=None, data_parallel=False):
        self.model.eval()
        self.load(model_file)

        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        loss_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device).float()
                aug1, aug2 = self._build_two_views(batch)
                with autocast(enabled=self.use_amp):
                    mod_features1, mod_features2 = model(aug1, aug2, proj_head=True)
                    loss = self.criterion(mod_features1, mod_features2).mean()

                loss_sum += loss.item()
                num_batches += 1

        return loss_sum / max(num_batches, 1)

    def load(self, model_file):
        if model_file:
            print("Loading the model from", model_file)
            checkpoint = torch.load(model_file + ".pt", map_location=self.device)
            checkpoint = adapt_focal_checkpoint(
                checkpoint,
                use_dual_modalities=getattr(self.model.backbone, "use_dual_modalities", True),
            )
            self.model.load_state_dict(checkpoint)

    def save(self, i=0):
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + ".pt")
        else:
            torch.save(self.model.state_dict(), self.save_path + ".pt")
