import copy
import time

import torch
import torch.nn as nn


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device

    def pretrain(self, func_loss, func_forward, func_evaluate,
                 data_loader_train, data_loader_test,
                 model_file=None, data_parallel=False, eval_batchwise=False):
        """Generic pretrain loop for both mask pretraining and contrastive pretraining."""
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        global_step = 0
        best_loss = 1e6
        model_best = copy.deepcopy(model.state_dict())

        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0
            time_sum = 0.0
            self.model.train()

            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]

                start_time = time.time()
                self.optimizer.zero_grad()

                loss = func_loss(model, batch)
                loss = loss.mean()  # for DataParallel compatibility

                loss.backward()
                self.optimizer.step()

                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return

            if eval_batchwise:
                loss_eva = self.run_batchwise(func_forward, func_evaluate, data_loader_test)
            else:
                loss_eva = self.run(func_forward, func_evaluate, data_loader_test)
            print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                  % (e + 1, self.cfg.n_epochs, loss_sum / len(data_loader_train), loss_eva))

            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)

        model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')

    def run(self, func_forward, func_evaluate, data_loader,
            model_file=None, data_parallel=False, load_self=False):
        """Evaluation Loop"""
        self.model.eval()
        self.load(model_file, load_self=load_self)

        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        results = []
        labels = []
        time_sum = 0.0

        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                start_time = time.time()
                result, label = func_forward(model, batch)
                time_sum += time.time() - start_time
                results.append(result)
                labels.append(label)

        if func_evaluate:
            return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
        else:
            return torch.cat(results, 0).cpu().numpy()

    def train(self, func_loss, func_forward, func_evaluate,
              data_loader_train, data_loader_test, data_loader_vali,
              model_file=None, data_parallel=False, load_self=False):
        """Supervised fine-tuning loop"""
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)

        global_step = 0
        vali_acc_best = 0.0
        best_stat = None
        model_best = copy.deepcopy(model.state_dict())

        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0
            time_sum = 0.0
            self.model.train()

            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return

            train_acc, train_f1 = self.run(func_forward, func_evaluate, data_loader_train)
            test_acc, test_f1 = self.run(func_forward, func_evaluate, data_loader_test)
            vali_acc, vali_f1 = self.run(func_forward, func_evaluate, data_loader_vali)

            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f'
                  % (e+1, self.cfg.n_epochs, loss_sum / len(data_loader_train),
                     train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1))

            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

    def load(self, model_file, load_self=False):
        if model_file:
            print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def save(self, i=0):
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(), self.save_path + '.pt')

    def run_batchwise(self, func_forward, func_evaluate, data_loader,
            model_file=None, data_parallel=False, load_self=False):
        """Batch-wise evaluation loop, suitable for contrastive losses."""
        self.model.eval()
        self.load(model_file, load_self=load_self)
    
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)
    
        metric_sum = 0.0
        num_batches = 0
    
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                result, label = func_forward(model, batch)
                metric = func_evaluate(label, result)
                metric_sum += float(metric)
                num_batches += 1
    
        return metric_sum / max(num_batches, 1)