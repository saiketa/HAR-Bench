import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss


class Trainer(object):
    """Trainer for TS-TCC self-supervised pretraining."""

    def __init__(
        self,
        cfg,
        model,
        temporal_contr_model,
        model_optimizer,
        temporal_optimizer,
        save_path,
        device,
    ):
        self.cfg = cfg
        self.model = model
        self.temporal_contr_model = temporal_contr_model
        self.model_optimizer = model_optimizer
        self.temporal_optimizer = temporal_optimizer
        self.save_path = save_path
        self.device = device

    def pretrain(
        self,
        data_loader_train,
        data_loader_vali,
        lambda1=1.0,
        lambda2=0.7,
        model_file=None,
        data_parallel=False,
    ):
        """TS-TCC self-supervised pretraining loop."""
        self.load(model_file)

        model = self.model.to(self.device)
        temporal_contr_model = self.temporal_contr_model.to(self.device)

        if data_parallel:
            model = nn.DataParallel(model)
            temporal_contr_model = nn.DataParallel(temporal_contr_model)

        best_loss = 1e6
        model_best = copy.deepcopy(model.state_dict())
        tc_best = copy.deepcopy(temporal_contr_model.state_dict())
        global_step = 0

        for e in range(self.cfg.n_epochs):
            model.train()
            temporal_contr_model.train()

            loss_sum = 0.0
            time_sum = 0.0

            for batch in data_loader_train:
                aug1, aug2 = batch
                aug1 = aug1.float().to(self.device)
                aug2 = aug2.float().to(self.device)

                start_time = time.time()

                self.model_optimizer.zero_grad()
                self.temporal_optimizer.zero_grad()

                # base model forward
                _, features1 = model(aug1)
                _, features2 = model(aug2)

                # normalize feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                # temporal contrastive branch
                temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)

                # NT-Xent branch
                nt_xent_criterion = NTXentLoss(
                    device=self.device,
                    batch_size=aug1.shape[0]
                )
                loss_nt = nt_xent_criterion(temp_cont_feat1, temp_cont_feat2)

                # total loss
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + loss_nt * lambda2
                loss = loss.mean()  # for DataParallel compatibility

                loss.backward()
                self.model_optimizer.step()
                self.temporal_optimizer.step()

                time_sum += time.time() - start_time
                loss_sum += loss.item()
                global_step += 1

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return

            loss_eva = self.evaluate(
                data_loader_vali,
                lambda1=lambda1,
                lambda2=lambda2,
                data_parallel=data_parallel
            )

            print(
                "Epoch %d/%d : Average Loss %5.4f. Vali Loss %5.4f"
                % (e + 1, self.cfg.n_epochs, loss_sum / len(data_loader_train), loss_eva)
            )

            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                tc_best = copy.deepcopy(temporal_contr_model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        self.temporal_contr_model.load_state_dict(tc_best)
        print("The Total Epoch have been reached.")

    def evaluate(
        self,
        data_loader,
        lambda1=1.0,
        lambda2=0.7,
        model_file=None,
        data_parallel=False,
    ):
        """Batch-wise evaluation for TS-TCC self-supervised loss."""
        self.model.eval()
        self.temporal_contr_model.eval()
        self.load(model_file)

        model = self.model.to(self.device)
        temporal_contr_model = self.temporal_contr_model.to(self.device)

        if data_parallel:
            model = nn.DataParallel(model)
            temporal_contr_model = nn.DataParallel(temporal_contr_model)

        loss_sum = 0.0
        num_batches = 0
        time_sum = 0.0

        with torch.no_grad():
            for batch in data_loader:
                aug1, aug2 = batch
                aug1 = aug1.float().to(self.device)
                aug2 = aug2.float().to(self.device)

                start_time = time.time()

                _, features1 = model(aug1)
                _, features2 = model(aug2)

                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)

                nt_xent_criterion = NTXentLoss(
                    device=self.device,
                    batch_size=aug1.shape[0]
                )
                loss_nt = nt_xent_criterion(temp_cont_feat1, temp_cont_feat2)

                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + loss_nt * lambda2
                loss = loss.mean()

                time_sum += time.time() - start_time
                loss_sum += loss.item()
                num_batches += 1

        return loss_sum / max(num_batches, 1)

    def load(self, model_file):
        """Load both base model and temporal contrastive model."""
        if model_file:
            print("Loading the model from", model_file)
            checkpoint = torch.load(model_file + ".pt", map_location=self.device)

            if "model_state_dict" in checkpoint and "temporal_contr_model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.temporal_contr_model.load_state_dict(checkpoint["temporal_contr_model_state_dict"])
            else:
                raise KeyError(
                    "Checkpoint must contain 'model_state_dict' and "
                    "'temporal_contr_model_state_dict'."
                )

    def save(self, i=0):
        """Save both base model and temporal contrastive model."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "temporal_contr_model_state_dict": self.temporal_contr_model.state_dict(),
        }

        if i != 0:
            torch.save(checkpoint, self.save_path + "_" + str(i) + ".pt")
        else:
            torch.save(checkpoint, self.save_path + ".pt")