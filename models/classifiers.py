import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierLSTM(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('lstm' + str(i), nn.LSTM(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('lstm' + str(i),
                                 nn.LSTM(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.seq_len))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            lstm = self.__getattr__('lstm' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h, _ = lstm(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierGRU(nn.Module):
    def __init__(self, cfg, input=None, output=None, feats=False):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i), nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i],
                                         batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierAttn(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.embd = nn.Embedding(cfg.seq_len, input)
        self.proj_q = nn.Linear(input, cfg.atten_hidden)
        self.proj_k = nn.Linear(input, cfg.atten_hidden)
        self.proj_v = nn.Linear(input, cfg.atten_hidden)
        self.attn = nn.MultiheadAttention(cfg.atten_hidden, cfg.num_head)
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.flatten = nn.Flatten()
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        seq_len = input_seqs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        pos = pos.unsqueeze(0).expand(input_seqs.size(0), seq_len)  # (S,) -> (B, S)
        h = input_seqs + self.embd(pos)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        h, weights = self.attn(q, k, v)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            if i == self.num_linear - 1:
                h = self.flatten(h)
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN2D(nn.Module):
    def __init__(self, cfg, output=None):
        super().__init__()
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i), nn.Conv2d(1, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i), nn.Conv2d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.BatchNorm2d(cfg.conv_io[i][1]))
        self.pool = nn.MaxPool2d(cfg.pool[0], stride=cfg.pool[1], padding=cfg.pool[2])
        self.flatten = nn.Flatten()
        for i in range(cfg.num_linear):
            if i == 0:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.flat_num, cfg.linear_io[i][1]))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = bn(self.pool(h))
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierCNN1D(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        in_channels = input if input is not None else cfg.input
        for i in range(cfg.num_cnn):
            if i == 0:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(in_channels, cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            else:
                self.__setattr__('cnn' + str(i),
                                 nn.Conv1d(cfg.conv_io[i][0], cfg.conv_io[i][1], cfg.conv_io[i][2], padding=cfg.conv_io[i][3]))
            self.__setattr__('bn' + str(i), nn.GroupNorm(1, cfg.conv_io[i][1]))

        pool_kernel = cfg.pool[0] if len(cfg.pool) > 0 else 2
        pool_stride = cfg.pool[1] if len(cfg.pool) > 1 else 2
        pool_padding = cfg.pool[2] if len(cfg.pool) > 2 else 0
        self.pool = nn.MaxPool1d(pool_kernel, stride=pool_stride, padding=pool_padding)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.pool_kernel = pool_kernel
        self.pool_padding = pool_padding

        for i in range(cfg.num_linear):
            if i == 0:
                in_dim = cfg.flat_num if cfg.flat_num > 0 else cfg.conv_io[-1][1]
                out_dim = cfg.linear_io[i][1] if cfg.linear_io[i][1] != 0 else output
                self.__setattr__('lin' + str(i), nn.Linear(in_dim, out_dim))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_cnn = cfg.num_cnn
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        # input_seqs: [B, T, C] -> conv over temporal dimension
        h = input_seqs.transpose(1, 2)
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = bn(h)
            # For sample-level embeddings such as CRT, temporal length can be 1.
            # Skip intermediate pooling when it would collapse the sequence to length 0.
            min_len_for_pool = max(int(self.pool_kernel) - 2 * int(self.pool_padding), 1)
            if h.size(-1) >= min_len_for_pool:
                h = self.pool(h)
        h = self.global_pool(h).squeeze(-1)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ and i != self.num_linear - 1:
                h = F.relu(h)
                if self.dropout:
                    h = F.dropout(h, training=training)
        return h


class BERTClassifier(nn.Module):

    def __init__(self, bert_cfg, classifier=None, frozen_bert=False):
        super().__init__()
        self.transformer = Transformer(bert_cfg)
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = classifier

    def forward(self, input_seqs, training=False): #, training
        h = self.transformer(input_seqs)
        h = self.classifier(h, training)
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)

class ClassifierMLP(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()

        # self.flatten_input = getattr(cfg, 'flatten', True)
        self.flatten_input = True
        for i in range(cfg.num_linear):
            if i == 0:
                out_dim = cfg.linear_io[i][1] if cfg.linear_io[i][1] != 0 else output
                # Use LazyLinear for the first projection so the MLP remains valid
                # when embedding sequence length differs from the static config
                # (for example CRT sample-level embeddings).
                self.__setattr__('lin' + str(i), nn.LazyLinear(out_dim))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                in_i = cfg.linear_io[i][0]
                out_i = cfg.linear_io[i][1]
                if out_i == 0 and output is not None:
                    out_i = output
                self.__setattr__('lin' + str(i), nn.Linear(in_i, out_i))

        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_linear = cfg.num_linear
        self.flatten = nn.Flatten()

    def forward(self, input_seqs, training=False):
        h = input_seqs

        if self.flatten_input:
            h = self.flatten(h)

        if self.dropout:
            h = F.dropout(h, training=training)

        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ and i != self.num_linear - 1:
                h = F.relu(h)
                if self.dropout:
                    h = F.dropout(h, training=training)
        return h

class ClassifierTransformer(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()

        self.input_proj = nn.Linear(input, cfg.hidden)

        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.num_head,
            dim_feedforward=cfg.ff_hidden,
            dropout=0.1 if cfg.dropout else 0.0,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers
        )

        for i in range(cfg.num_linear):
            if i == 0:
                in_dim = cfg.hidden
                out_dim = cfg.linear_io[i][1] if cfg.linear_io[i][1] != 0 else output
                self.__setattr__('lin' + str(i), nn.Linear(in_dim, out_dim))
            elif output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                in_i = cfg.linear_io[i][0]
                out_i = cfg.linear_io[i][1]
                if out_i == 0 and output is not None:
                    out_i = output
                self.__setattr__('lin' + str(i), nn.Linear(in_i, out_i))

        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_linear = cfg.num_linear

    def forward(self, input_seqs, training=False):
        # input_seqs: [B, T, C]
        bsz, seq_len, _ = input_seqs.size()
        pos = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        pos = pos.unsqueeze(0).expand(bsz, seq_len)

        h = self.input_proj(input_seqs) + self.pos_emb(pos)
        h = self.encoder(h)

        # mean pooling over temporal dimension
        h = h.mean(dim=1)

        if self.dropout:
            h = F.dropout(h, training=training)

        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ and i != self.num_linear - 1:
                h = F.relu(h)
                if self.dropout:
                    h = F.dropout(h, training=training)
        return h

def fetch_classifier(method, model_cfg, input=None, output=None, feats=False):
    if 'lstm' in method:
        model = ClassifierLSTM(model_cfg, input=input, output=output)
    elif 'gru' in method:
        model = ClassifierGRU(model_cfg, input=input, output=output)
    elif 'mlp' in method:
        model = ClassifierMLP(model_cfg, input=input, output=output)
    elif 'transformer' in method:
        model = ClassifierTransformer(model_cfg, input=input, output=output)
    elif 'dcnn' in method:
        model = BenchmarkDCNN(model_cfg, input=input, output=output)
    elif 'cnn2' in method:
        model = ClassifierCNN2D(model_cfg, output=output)
    elif method == 'cnn' or 'cnn1' in method:
        model = ClassifierCNN1D(model_cfg, input=input, output=output)
    elif 'deepsense' in method:
        model = BenchmarkDeepSense(model_cfg, input=input, output=output)
    elif 'attn' in method:
        model = ClassifierAttn(model_cfg, input=input, output=output)
    else:
        model = None
    return model
