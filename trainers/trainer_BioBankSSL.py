import copy
import time

import torch
import torch.nn as nn


class Trainer(object):
    """Trainer for BioBankSSL multi-task self-supervised pretraining."""

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        save_path,
        device,
        criterion=None,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def _compute_acc(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        return acc

    def _compute_loss(
        self,
        aot_pred,
        permute_pred,
        time_w_pred,
        aot_y,
        permute_y,
        time_w_y,
        lambda_aot=1.0,
        lambda_permute=1.0,
        lambda_time_w=1.0,
    ):
        loss_aot = self.criterion(aot_pred, aot_y)
        loss_permute = self.criterion(permute_pred, permute_y)
        loss_time_w = self.criterion(time_w_pred, time_w_y)

        loss = (
            lambda_aot * loss_aot
            + lambda_permute * loss_permute
            + lambda_time_w * loss_time_w
        ) / 3.0

        acc_aot = self._compute_acc(aot_pred, aot_y)
        acc_permute = self._compute_acc(permute_pred, permute_y)
        acc_time_w = self._compute_acc(time_w_pred, time_w_y)
        acc_mean = (acc_aot + acc_permute + acc_time_w) / 3.0

        metrics = {
            "loss": loss,
            "loss_aot": loss_aot,
            "loss_permute": loss_permute,
            "loss_time_w": loss_time_w,
            "acc_aot": acc_aot,
            "acc_permute": acc_permute,
            "acc_time_w": acc_time_w,
            "acc_mean": acc_mean,
        }
        return metrics

    def pretrain(
        self,
        data_loader_train,
        data_loader_vali,
        lambda_aot=1.0,
        lambda_permute=1.0,
        lambda_time_w=1.0,
        model_file=None,
        data_parallel=False,
    ):
        """BioBankSSL multi-task self-supervised pretraining loop."""
        self.load(model_file)

        model = self.model.to(self.device)

        if data_parallel:
            model = nn.DataParallel(model)

        best_loss = 1e6
        model_best = copy.deepcopy(model.state_dict())
        global_step = 0

        for e in range(self.cfg.n_epochs):
            model.train()

            loss_sum = 0.0
            loss_aot_sum = 0.0
            loss_permute_sum = 0.0
            loss_time_w_sum = 0.0

            acc_aot_sum = 0.0
            acc_permute_sum = 0.0
            acc_time_w_sum = 0.0
            acc_mean_sum = 0.0

            time_sum = 0.0
            num_batches = 0

            for batch in data_loader_train:
                x, aot_y, permute_y, time_w_y = batch

                x = x.float().to(self.device)
                aot_y = aot_y.long().to(self.device)
                permute_y = permute_y.long().to(self.device)
                time_w_y = time_w_y.long().to(self.device)

                start_time = time.time()

                self.optimizer.zero_grad()

                aot_pred, permute_pred, time_w_pred, _ = model(x)

                metrics = self._compute_loss(
                    aot_pred=aot_pred,
                    permute_pred=permute_pred,
                    time_w_pred=time_w_pred,
                    aot_y=aot_y,
                    permute_y=permute_y,
                    time_w_y=time_w_y,
                    lambda_aot=lambda_aot,
                    lambda_permute=lambda_permute,
                    lambda_time_w=lambda_time_w,
                )

                loss = metrics["loss"]
                loss = loss.mean()  # for DataParallel compatibility

                loss.backward()
                self.optimizer.step()

                time_sum += time.time() - start_time

                loss_sum += loss.item()
                loss_aot_sum += metrics["loss_aot"].mean().item()
                loss_permute_sum += metrics["loss_permute"].mean().item()
                loss_time_w_sum += metrics["loss_time_w"].mean().item()

                acc_aot_sum += metrics["acc_aot"].mean().item()
                acc_permute_sum += metrics["acc_permute"].mean().item()
                acc_time_w_sum += metrics["acc_time_w"].mean().item()
                acc_mean_sum += metrics["acc_mean"].mean().item()

                num_batches += 1
                global_step += 1

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return

            eval_metrics = self.evaluate(
                data_loader=data_loader_vali,
                lambda_aot=lambda_aot,
                lambda_permute=lambda_permute,
                lambda_time_w=lambda_time_w,
                data_parallel=data_parallel,
            )

            print(
                "Epoch %d/%d : "
                "Train Loss %5.4f (aot %5.4f, perm %5.4f, tw %5.4f), "
                "Train Acc %5.4f (aot %5.4f, perm %5.4f, tw %5.4f), "
                "Vali Loss %5.4f, Vali Acc %5.4f"
                % (
                    e + 1,
                    self.cfg.n_epochs,
                    loss_sum / max(num_batches, 1),
                    loss_aot_sum / max(num_batches, 1),
                    loss_permute_sum / max(num_batches, 1),
                    loss_time_w_sum / max(num_batches, 1),
                    acc_mean_sum / max(num_batches, 1),
                    acc_aot_sum / max(num_batches, 1),
                    acc_permute_sum / max(num_batches, 1),
                    acc_time_w_sum / max(num_batches, 1),
                    eval_metrics["loss"],
                    eval_metrics["acc_mean"],
                )
            )

            if eval_metrics["loss"] < best_loss:
                best_loss = eval_metrics["loss"]
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        print("The Total Epoch have been reached.")

    def evaluate(
        self,
        data_loader,
        lambda_aot=1.0,
        lambda_permute=1.0,
        lambda_time_w=1.0,
        model_file=None,
        data_parallel=False,
    ):
        """Batch-wise evaluation for BioBankSSL self-supervised loss."""
        self.model.eval()
        self.load(model_file)

        model = self.model.to(self.device)

        if data_parallel:
            model = nn.DataParallel(model)

        loss_sum = 0.0
        loss_aot_sum = 0.0
        loss_permute_sum = 0.0
        loss_time_w_sum = 0.0

        acc_aot_sum = 0.0
        acc_permute_sum = 0.0
        acc_time_w_sum = 0.0
        acc_mean_sum = 0.0

        num_batches = 0
        time_sum = 0.0

        with torch.no_grad():
            for batch in data_loader:
                x, aot_y, permute_y, time_w_y = batch

                x = x.float().to(self.device)
                aot_y = aot_y.long().to(self.device)
                permute_y = permute_y.long().to(self.device)
                time_w_y = time_w_y.long().to(self.device)

                start_time = time.time()

                aot_pred, permute_pred, time_w_pred, _ = model(x)

                metrics = self._compute_loss(
                    aot_pred=aot_pred,
                    permute_pred=permute_pred,
                    time_w_pred=time_w_pred,
                    aot_y=aot_y,
                    permute_y=permute_y,
                    time_w_y=time_w_y,
                    lambda_aot=lambda_aot,
                    lambda_permute=lambda_permute,
                    lambda_time_w=lambda_time_w,
                )

                loss = metrics["loss"]
                loss = loss.mean()

                time_sum += time.time() - start_time

                loss_sum += loss.item()
                loss_aot_sum += metrics["loss_aot"].mean().item()
                loss_permute_sum += metrics["loss_permute"].mean().item()
                loss_time_w_sum += metrics["loss_time_w"].mean().item()

                acc_aot_sum += metrics["acc_aot"].mean().item()
                acc_permute_sum += metrics["acc_permute"].mean().item()
                acc_time_w_sum += metrics["acc_time_w"].mean().item()
                acc_mean_sum += metrics["acc_mean"].mean().item()

                num_batches += 1

        results = {
            "loss": loss_sum / max(num_batches, 1),
            "loss_aot": loss_aot_sum / max(num_batches, 1),
            "loss_permute": loss_permute_sum / max(num_batches, 1),
            "loss_time_w": loss_time_w_sum / max(num_batches, 1),
            "acc_aot": acc_aot_sum / max(num_batches, 1),
            "acc_permute": acc_permute_sum / max(num_batches, 1),
            "acc_time_w": acc_time_w_sum / max(num_batches, 1),
            "acc_mean": acc_mean_sum / max(num_batches, 1),
        }
        return results

    def load(self, model_file):
        """Load BioBankSSL base model."""
        if model_file:
            print("Loading the model from", model_file)
            checkpoint = torch.load(model_file + ".pt", map_location=self.device)

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise KeyError("Checkpoint must contain 'model_state_dict'.")

    def save(self, i=0):
        """Save BioBankSSL base model."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }

        if i != 0:
            torch.save(checkpoint, self.save_path + "_" + str(i) + ".pt")
        else:
            torch.save(checkpoint, self.save_path + ".pt")
