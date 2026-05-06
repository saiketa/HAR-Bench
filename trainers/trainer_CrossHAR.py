import time
import torch

from models.loss import NTXentLoss


class Trainer(object):
    """Training Helper Class for CrossHAR pretraining"""

    def __init__(
        self,
        cfg,
        masked_model,
        masked_optimizer,
        contrastive_model,
        contrastive_optimizer,
        save_path,
        device,
        criterion,
    ):
        self.cfg = cfg
        self.masked_model = masked_model
        self.masked_optimizer = masked_optimizer
        self.contrastive_model = contrastive_model
        self.contrastive_optimizer = contrastive_optimizer
        self.save_path = save_path
        self.device = device
        self.criterion = criterion

    def pretrain(
        self,
        data_loader_train,
        data_loader_vali,
        lambda1=6.0,
        lambda2=1.0,
        model_file=None,
    ):
        print(f'CrossHAR pretrain | mlm loss : nt loss = {lambda1} : {lambda2}')

        n_epoch_now = 0
        global_step = 0
        best_loss = 1e6

        self.load(model_file, lambda1=lambda1, lambda2=lambda2)

        self.masked_model = self.masked_model.to(self.device)
        self.contrastive_model = self.contrastive_model.to(self.device)

        nt_xent_criterion = NTXentLoss(
            device=self.device,
            batch_size=self.cfg.batch_size
        )

        for e in range(n_epoch_now, self.cfg.n_epochs):
            self.masked_model.train()
            self.contrastive_model.train()

            loss_sum = 0.0
            loss_mlm_sum = 0.0
            loss_nt_sum = 0.0
            time_sum = 0.0

            for i, batch in enumerate(data_loader_train):
                start_time = time.time()

                (
                    mask_seqs_1, masked_pos_1, seqs_1,
                    mask_seqs_2, masked_pos_2, seqs_2
                ) = batch

                mask_seqs_1 = mask_seqs_1.to(self.device)
                masked_pos_1 = masked_pos_1.to(self.device)
                seqs_1 = seqs_1.to(self.device)

                mask_seqs_2 = mask_seqs_2.to(self.device)
                masked_pos_2 = masked_pos_2.to(self.device)
                seqs_2 = seqs_2.to(self.device)

                self.masked_optimizer.zero_grad()
                self.contrastive_optimizer.zero_grad()

                representation_1, seq_recon_1 = self.masked_model(mask_seqs_1, masked_pos_1)
                loss_lm_1 = self.criterion(seq_recon_1, seqs_1).mean()

                representation_2, seq_recon_2 = self.masked_model(mask_seqs_2, masked_pos_2)
                loss_lm_2 = self.criterion(seq_recon_2, seqs_2).mean()

                zis = self.contrastive_model(representation_1)
                zjs = self.contrastive_model(representation_2)
                loss_nt = nt_xent_criterion(zis, zjs)

                loss_mlm = (loss_lm_1 + loss_lm_2) / 2.0

                if e < (self.cfg.n_epochs - self.cfg.n_epochs_cl):
                    loss = loss_mlm
                else:
                    if e == (self.cfg.n_epochs - self.cfg.n_epochs_cl):
                        best_loss = 1e6
                    loss = lambda1 * loss_mlm + lambda2 * loss_nt

                loss.backward()
                self.masked_optimizer.step()
                self.contrastive_optimizer.step()

                time_sum += time.time() - start_time
                global_step += 1

                loss_sum += loss.item()
                loss_mlm_sum += loss_mlm.item()
                loss_nt_sum += loss_nt.item()

                if self.cfg.total_steps and global_step >= self.cfg.total_steps:
                    print('The total training steps have been reached.')
                    return

            train_len = len(data_loader_train)
            train_loss = loss_sum / train_len
            train_loss_mlm = loss_mlm_sum / train_len
            train_loss_nt = loss_nt_sum / train_len

            val_loss, val_loss_mlm, val_loss_nt = self.run(
                data_loader=data_loader_vali,
                epoch=e,
                lambda1=lambda1,
                lambda2=lambda2,
                nt_xent_criterion=nt_xent_criterion
            )

            print(
                'Epoch %d/%d : Train Loss %5.4f | mlm %5.4f | nt %5.4f || '
                'Val Loss %5.4f | mlm %5.4f | nt %5.4f'
                % (
                    e + 1,
                    self.cfg.n_epochs,
                    train_loss,
                    train_loss_mlm,
                    train_loss_nt,
                    val_loss,
                    val_loss_mlm,
                    val_loss_nt
                )
            )

            if val_loss < best_loss:
                best_loss = val_loss
                self.save(lambda1=lambda1, lambda2=lambda2)

        print('The total epochs have been reached.')

    def run(
        self,
        data_loader,
        epoch,
        lambda1,
        lambda2,
        nt_xent_criterion,
    ):
        """Evaluation loop"""
        self.masked_model.eval()
        self.contrastive_model.eval()

        loss_sum = 0.0
        loss_mlm_sum = 0.0
        loss_nt_sum = 0.0

        with torch.no_grad():
            for batch in data_loader:
                (
                    mask_seqs_1, masked_pos_1, seqs_1,
                    mask_seqs_2, masked_pos_2, seqs_2
                ) = batch

                mask_seqs_1 = mask_seqs_1.to(self.device)
                masked_pos_1 = masked_pos_1.to(self.device)
                seqs_1 = seqs_1.to(self.device)

                mask_seqs_2 = mask_seqs_2.to(self.device)
                masked_pos_2 = masked_pos_2.to(self.device)
                seqs_2 = seqs_2.to(self.device)

                representation_1, seq_recon_1 = self.masked_model(mask_seqs_1, masked_pos_1)
                loss_lm_1 = self.criterion(seq_recon_1, seqs_1).mean()

                representation_2, seq_recon_2 = self.masked_model(mask_seqs_2, masked_pos_2)
                loss_lm_2 = self.criterion(seq_recon_2, seqs_2).mean()

                zis = self.contrastive_model(representation_1)
                zjs = self.contrastive_model(representation_2)
                loss_nt = nt_xent_criterion(zis, zjs)

                loss_mlm = (loss_lm_1 + loss_lm_2) / 2.0

                if epoch < (self.cfg.n_epochs - self.cfg.n_epochs_cl):
                    loss = loss_mlm
                else:
                    loss = lambda1 * loss_mlm + lambda2 * loss_nt

                loss_sum += loss.item()
                loss_mlm_sum += loss_mlm.item()
                loss_nt_sum += loss_nt.item()

        data_len = len(data_loader)
        return (
            loss_sum / data_len,
            loss_mlm_sum / data_len,
            loss_nt_sum / data_len,
        )

    def load(self, model_file=None, lambda1=6.0, lambda2=1.0, load_self=False):
        """Load saved model or pretrained weights"""
        if model_file is None:
            return

        if load_self:
            self.masked_model.load_self(model_file + '.pt', map_location=self.device)
            return

        masked_model_path = f'{model_file}_masked_{lambda1}_{lambda2}.pt'
        contrastive_model_path = f'{model_file}_contrastive_{lambda1}_{lambda2}.pt'

        print('Loading pretrained masked model from', masked_model_path)
        print('Loading pretrained contrastive model from', contrastive_model_path)

        self.masked_model.load_state_dict(
            torch.load(masked_model_path, map_location=self.device)
        )
        self.contrastive_model.load_state_dict(
            torch.load(contrastive_model_path, map_location=self.device)
        )

    def save(self, lambda1=6.0, lambda2=1.0):
        """Save current model"""
        masked_model_path = f'{self.save_path}_masked_{lambda1}_{lambda2}.pt'
        contrastive_model_path = f'{self.save_path}_contrastive_{lambda1}_{lambda2}.pt'

        torch.save(self.masked_model.state_dict(), masked_model_path)
        torch.save(self.contrastive_model.state_dict(), contrastive_model_path)