import copy

import torch
import torch.nn as nn

from models.SimMTM import data_transform_masked4cl


class Trainer(object):
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device
        self.masking_ratio = float(getattr(cfg, "simmtm_masking_ratio", 0.5))
        self.lm = int(getattr(cfg, "simmtm_lm", 3))
        positive_nums = int(getattr(cfg, "simmtm_positive_nums", 3))
        self.positive_nums = positive_nums if positive_nums > 0 else None

    def _compute_loss(self, x):
        data_masked_m, _ = data_transform_masked4cl(
            x,
            masking_ratio=self.masking_ratio,
            lm=self.lm,
            positive_nums=self.positive_nums,
        )
        data_masked_om = torch.cat([x, data_masked_m.to(x.device)], dim=0)
        out = self.model(data_masked_om, pretrain=True)
        return out

    def pretrain(self, data_loader_train, data_loader_vali, model_file=None, data_parallel=False):
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        best_loss = 1e6
        model_best = copy.deepcopy(self.model.state_dict())
        global_step = 0

        for e in range(self.cfg.n_epochs):
            self.model.train()
            loss_sum = 0.0
            loss_cl_sum = 0.0
            loss_rb_sum = 0.0
            num_batches = 0

            for batch in data_loader_train:
                x = batch.to(self.device).float()
                self.optimizer.zero_grad()

                out = self._compute_loss(x)
                loss = out["loss"].mean()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                loss_cl_sum += out["loss_cl"].mean().item()
                loss_rb_sum += out["loss_rb"].mean().item()
                num_batches += 1
                global_step += 1

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return

            eval_metrics = self.evaluate(data_loader_vali, data_parallel=data_parallel)
            print(
                "Epoch %d/%d : Train Loss %5.4f (cl %5.4f, rb %5.4f), Vali Loss %5.4f"
                % (
                    e + 1,
                    self.cfg.n_epochs,
                    loss_sum / max(num_batches, 1),
                    loss_cl_sum / max(num_batches, 1),
                    loss_rb_sum / max(num_batches, 1),
                    eval_metrics["loss"],
                )
            )

            if eval_metrics["loss"] < best_loss:
                best_loss = eval_metrics["loss"]
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

        loss_sum = 0.0
        loss_cl_sum = 0.0
        loss_rb_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(self.device).float()
                out = self._compute_loss(x)
                loss_sum += out["loss"].mean().item()
                loss_cl_sum += out["loss_cl"].mean().item()
                loss_rb_sum += out["loss_rb"].mean().item()
                num_batches += 1

        return {
            "loss": loss_sum / max(num_batches, 1),
            "loss_cl": loss_cl_sum / max(num_batches, 1),
            "loss_rb": loss_rb_sum / max(num_batches, 1),
        }

    def load(self, model_file):
        if model_file:
            print("Loading the model from", model_file)
            checkpoint = torch.load(model_file + ".pt", map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

    def save(self, i=0):
        checkpoint = {"model_state_dict": self.model.state_dict()}
        if i != 0:
            torch.save(checkpoint, self.save_path + "_" + str(i) + ".pt")
        else:
            torch.save(checkpoint, self.save_path + ".pt")
