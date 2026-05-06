# trainers/trainer_CRT.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer(object):
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device

    def pretrain(
        self,
        data_loader_train,
        data_loader_vali=None,
        min_mask_ratio=0.3,
        max_mask_ratio=0.8,
        beta=1e-4,
        model_file=None,
        data_parallel=False,
    ):
        self.load(model_file)

        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=20
        )

        best_loss = 1e9
        model_best = copy.deepcopy(model.state_dict())
        global_step = 0

        for e in range(self.cfg.n_epochs):
            model.train()
            ratio = max(min_mask_ratio, min(max_mask_ratio, float(e) / max(self.cfg.n_epochs, 1)))

            loss_sum = 0.0
            recon_sum = 0.0
            idc_sum = 0.0
            num_batches = 0

            for batch in data_loader_train:
                x = batch.float().to(self.device)

                self.optimizer.zero_grad()
                out = model(x, mask_ratio=ratio, beta=beta)

                loss = out["loss"].mean()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                recon_sum += out["recon_loss"].mean().item()
                idc_sum += out["idc_loss"].mean().item()
                num_batches += 1
                global_step += 1

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return

            # 与原始逻辑保持一致（按 epoch index 调度）
            scheduler.step(e)

            train_loss = loss_sum / max(num_batches, 1)
            print(
                "Epoch %d/%d : Train Loss %5.4f (recon %5.4f, idc %5.4f), mask_ratio %4.3f"
                % (
                    e + 1,
                    self.cfg.n_epochs,
                    train_loss,
                    recon_sum / max(num_batches, 1),
                    idc_sum / max(num_batches, 1),
                    ratio,
                )
            )

            if train_loss < best_loss:
                best_loss = train_loss
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        print("The Total Epoch have been reached.")

    def load(self, model_file):
        if model_file:
            print("Loading the model from", model_file)
            checkpoint = torch.load(model_file + ".pt", map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise KeyError("Checkpoint must contain 'model_state_dict'.")

    def save(self, i=0):
        checkpoint = {"model_state_dict": self.model.state_dict()}
        if i != 0:
            torch.save(checkpoint, self.save_path + "_" + str(i) + ".pt")
        else:
            torch.save(checkpoint, self.save_path + ".pt")
