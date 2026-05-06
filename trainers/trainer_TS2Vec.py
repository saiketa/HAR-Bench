import copy
import time

import numpy as np
import torch
import torch.nn as nn

from models.TS2Vec import hierarchical_contrastive_loss


def take_per_row(a, start, length):
    all_indx = start[:, None] + torch.arange(length, device=a.device)
    return a[torch.arange(a.shape[0])[:, None], all_indx]


class Trainer(object):
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device
        self.temporal_unit = int(getattr(cfg, "ts2vec_temporal_unit", 0))
        self.max_train_length = int(getattr(cfg, "ts2vec_max_train_length", 0))
        self.alpha = float(getattr(cfg, "ts2vec_alpha", 0.5))

    def _compute_loss(self, model, x):
        if self.max_train_length > 0 and x.size(1) > self.max_train_length:
            window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
            x = x[:, window_offset : window_offset + self.max_train_length]

        ts_len = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_len + 1)
        crop_left = np.random.randint(ts_len - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_len + 1)
        crop_offset = np.random.randint(
            low=-crop_eleft,
            high=ts_len - crop_eright + 1,
            size=x.size(0),
        )
        crop_offset = torch.as_tensor(crop_offset, device=x.device)

        out1 = model(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
        out1 = out1[:, -crop_l:]

        out2 = model(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
        out2 = out2[:, :crop_l]

        return hierarchical_contrastive_loss(
            out1,
            out2,
            alpha=self.alpha,
            temporal_unit=self.temporal_unit,
        )

    def pretrain(self, data_loader_train, data_loader_vali, model_file=None, data_parallel=False):
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        global_step = 0
        best_loss = 1e6
        model_best = copy.deepcopy(self.model.state_dict())

        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0
            self.model.train()

            for batch in data_loader_train:
                x = batch.to(self.device).float()
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = self._compute_loss(model, x)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.model.net.update_parameters(self.model._net)

                global_step += 1
                loss_sum += loss.item()

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return

            loss_eva = self.evaluate(data_loader_vali, data_parallel=data_parallel)
            print(
                "Epoch %d/%d : Average Loss %5.4f. Vali Loss %5.4f"
                % (e + 1, self.cfg.n_epochs, loss_sum / len(data_loader_train), loss_eva)
            )

            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(self.model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        print("The Total Epoch have been reached.")

    def evaluate(self, data_loader, model_file=None, data_parallel=False):
        self.model.eval()
        self.load(model_file)

        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        metric_sum = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(self.device).float()
                loss = self._compute_loss(model, x)
                metric_sum += float(loss.mean().item())
                num_batches += 1
        return metric_sum / max(num_batches, 1)

    def load(self, model_file):
        if model_file:
            print("Loading the model from", model_file)
            checkpoint = torch.load(model_file + ".pt", map_location=self.device)
            if isinstance(checkpoint, dict) and "_net" in checkpoint and "net" in checkpoint:
                self.model._net.load_state_dict(checkpoint["_net"])
                self.model.net.load_state_dict(checkpoint["net"])
            else:
                self.model.load_state_dict(checkpoint)

    def save(self, i=0):
        checkpoint = {
            "_net": self.model._net.state_dict(),
            "net": self.model.net.state_dict(),
        }
        if i != 0:
            torch.save(checkpoint, self.save_path + "_" + str(i) + ".pt")
        else:
            torch.save(checkpoint, self.save_path + ".pt")
