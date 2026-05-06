# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/1/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : config.py
# @Description :

import json
from typing import NamedTuple
import os
# from bunch import bunchify


class PretrainModelConfig(NamedTuple):
    "Configuration for pretraining models"
    # Transformer/CNN backbone shared by pretraining models
    encoder_type: str = "transformer"
    hidden: int = 0
    hidden_ff: int = 0
    feature_num: int = 0
    n_layers: int = 0
    n_heads: int = 0
    seq_len: int = 0
    emb_norm: bool = True
    resnet_blocks: int = 4
    resnet_kernel_size: int = 5
    resnet_dropout: float = 0.1
    resnet_dilations: list = [1, 2, 4, 8]
    cnn_depth: int = 4
    cnn_kernel_size: int = 5
    cnn_dropout: float = 0.1
    cnn_dilations: list = [1, 2, 4, 8]
    cnn_hidden_multiplier: int = 2

    proj_dim: int = 0
    dropout: float = 0.1

    patch_len: int = 0
    focal_use_dual_modalities: bool = True
    @classmethod
    def from_json(cls, js):
        valid = cls._fields
        return cls(**{k: v for k, v in js.items() if k in valid})


class ClassifierModelConfig(NamedTuple):
    "Configuration for classifier model"
    seq_len: int = 0
    input: int = 0

    num_rnn: int = 0
    num_layers: int = 0
    rnn_io: list = []

    num_cnn: int = 0
    conv_io: list = []
    pool: list = []
    flat_num: int = 0

    num_attn: int = 0
    num_head: int = 0
    atten_hidden: int = 0
    hidden: int = 0
    ff_hidden: int = 0

    num_linear: int = 0
    linear_io: list = []

    activ: bool = False
    dropout: bool = False

    @classmethod
    def from_json(cls, js):
        return cls(**js)


# class TrainConfig(NamedTuple):
#     seed: int = 0
#     batch_size: int = 0
#     lr: float = 0.0
#     n_epochs: int = 0
#     n_epochs_cl: int = 0
#     warmup: float = 0.0
#     save_steps: int = 0
#     total_steps: int = 0
#     lambda1: float = 0.0
#     lambda2: float = 0.0
#     @classmethod
#     def from_json(cls, file):
#         return cls(**json.load(open(file, "r")))

class TrainConfig(NamedTuple):
    seed: int = 0
    batch_size: int = 0
    lr: float = 0.0
    n_epochs: int = 0
    n_epochs_cl: int = 0
    warmup: float = 0.0
    save_steps: int = 0
    total_steps: int = 0
    lambda1: float = 0.0
    lambda2: float = 0.0

    # FOCAL
    focal_temperature: float = 0.07
    focal_inter_rank_margin: float = 1.0
    focal_shared_contrastive_loss_weight: float = 1.0
    focal_private_contrastive_loss_weight: float = 1.0
    focal_orthogonal_loss_weight: float = 3.0
    focal_rank_loss_weight: float = 5.0
    focal_no_private: bool = False
    focal_seq_len: int = 4
    focal_eval_interval: int = 10
    focal_time_aug_prob: float = 0.5
    focal_phase_shift_prob: float = 0.5
    focal_scaling_sigma: float = 0.2
    focal_time_warp_sigma: float = 0.2
    focal_time_warp_knots: int = 6
    focal_mag_warp_sigma: float = 0.05
    focal_mag_warp_knots: int = 4
    focal_amp: bool = True

    # TS2Vec
    ts2vec_temporal_unit: int = 0
    ts2vec_max_train_length: int = 0
    ts2vec_alpha: float = 0.5

    # SimMTM
    simmtm_masking_ratio: float = 0.5
    simmtm_positive_nums: int = 3
    simmtm_lm: int = 3
    simmtm_temperature: float = 0.2

    @classmethod
    def from_json(cls, file):
        js = json.load(open(file, "r")) if isinstance(file, str) else file
        if "train" in js and not any(k in js for k in cls._fields):
            js = js["train"]
        valid = cls._fields
        return cls(**{k: v for k, v in js.items() if k in valid})


class MaskConfig(NamedTuple):
    """ Hyperparameters for training """
    mask_ratio: float = 0  # masking probability
    mask_alpha: int = 0  # How many tokens to form a group.
    max_gram: int = 0  # number of max n-gram to masking
    mask_prob: float = 1.0
    replace_prob: float = 0.0

    @classmethod
    def from_json(cls, source): # load config from json file or dict
        if isinstance(source, str):
            source = json.load(open(source, "r"))
        valid = cls._fields
        return cls(**{k: v for k, v in source.items() if k in valid})


class DatasetConfig(NamedTuple):
    """ Hyperparameters for training """
    sr: int = 0  # sampling rate
    # dataset = Narray with shape (size, seq_len, dimension)
    size: int = 0  # data sample number
    seq_len: int = 0  # seq length
    dimension: int = 0  # feature dimension

    activity_label_index: int = -1  # index of activity label
    activity_label_size: int = 0  # number of activity label
    activity_label: list = []  # names of activity label.

    user_label_index: int = -1  # index of user label
    user_label_size: int = 0  # number of user label

    position_label_index: int = -1  # index of phone position label
    position_label_size: int = 0  # number of position label
    position_label: list = []  # names of position label.

    model_label_index: int = -1  # index of phone model label
    model_label_size: int = 0  # number of model label

    @classmethod
    def from_json(cls, js):
        return cls(**js)


def create_io_config(args, dataset_name, version, pretrain_model=None, target='pretrain'):
    data_path = os.path.join('dataset', dataset_name, 'data_' + version + '.npy')
    label_path = os.path.join('dataset', dataset_name, 'label_' + version + '.npy')
    args.data_path = data_path
    args.label_path = label_path

    save_path = os.path.join('saved', target + "_" + dataset_name + "_" + version)  # + "_temp"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    args.save_path = os.path.join(save_path, args.save_model)

    # log_path = os.path.join('log', target + "_" + dataset_name + "_" + version)  # + "_temp"
    # if not os.path.exists(log_path):
    #     os.mkdir(log_path)
    # args.log_dir = log_path

    if pretrain_model is not None:
        if target.count('_') > 2: # bert_classifier
            model_path = os.path.join('saved', 'pretrain_' + target.split('_')[2] + "_" + dataset_name + "_" + version, pretrain_model)
        else:
            model_path = os.path.join(save_path, pretrain_model)
        args.pretrain_model = model_path
    else:
        args.pretrain_model = None
    return args

def load_method_config(prefix, path_method_dir='config'):
    path = os.path.join(path_method_dir, f'{prefix}.json')
    if not os.path.exists(path):
        return {}
    return json.load(open(path, "r"))


def load_mask_config(prefix, path_method_dir='config'):
    method_cfg = load_method_config(prefix, path_method_dir=path_method_dir)
    mask_keys = set(MaskConfig._fields)
    if any(key in method_cfg for key in mask_keys):
        return MaskConfig.from_json(method_cfg)

    return MaskConfig()


def load_encoder_config(encoder_backbone='transformer', path_encoder='config/encoder.json'):
    if not os.path.exists(path_encoder):
        return None
    encoder_config_all = json.load(open(path_encoder, "r"))
    encoder_backbone = str(encoder_backbone).lower()
    return encoder_config_all.get(encoder_backbone)


def load_pretrain_model_config(prefix, encoder_backbone='transformer',
                               path_encoder='config/encoder.json',
                               path_method_dir='config',
                               legacy_path_model=None):
    encoder_cfg = load_encoder_config(encoder_backbone, path_encoder=path_encoder)
    if encoder_cfg is not None:
        method_path = os.path.join(path_method_dir, f'{prefix}.json')
        if not os.path.exists(method_path):
            return None
        merged = dict(encoder_cfg)
        merged.update(load_method_config(prefix, path_method_dir=path_method_dir))
        merged["encoder_type"] = str(encoder_backbone).lower()
        return PretrainModelConfig.from_json(merged)

    if legacy_path_model and os.path.exists(legacy_path_model):
        model_config_all = json.load(open(legacy_path_model, "r"))
        if prefix in model_config_all:
            return PretrainModelConfig.from_json(model_config_all[prefix])
    return None


def load_model_config(target, prefix, version,
                      path_model=None,
                      path_classifier='config/classifier.json',
                      path_encoder='config/encoder.json',
                      path_method_dir='config',
                      encoder_backbone=None):
    if "bert" not in target: # pretrain or pure classifier
        if "pretrain" in target:
            return load_pretrain_model_config(
                prefix,
                encoder_backbone=encoder_backbone or "transformer",
                path_encoder=path_encoder,
                path_method_dir=path_method_dir,
                legacy_path_model=path_model,
            )
        else:
            model_config_all = json.load(open(path_classifier, "r"))
        name = prefix
        if name in model_config_all:
            if "pretrain" in target:
                return PretrainModelConfig.from_json(model_config_all[name])
            else:
                return ClassifierModelConfig.from_json(model_config_all[name])
        else:
            return None
    else: # pretrain + classifier for fine-tune
        model_config_classifier = json.load(open(path_classifier, "r"))
        prefixes = prefix.split('_')
        versions = version.split('_')
        bert_name = prefixes[0] + "_" + versions[0]
        classifier_name = prefixes[1] + "_" + versions[1]
        bert_cfg = load_pretrain_model_config(
            bert_name,
            encoder_backbone=encoder_backbone or "transformer",
            path_encoder=path_encoder,
            path_method_dir=path_method_dir,
            legacy_path_model=path_model,
        )
        if bert_cfg is not None and classifier_name in model_config_classifier:
            return [bert_cfg, ClassifierModelConfig.from_json(model_config_classifier[classifier_name])]
        else:
            return None


def load_dataset_stats(dataset, version):
    path = 'dataset/data_config.json'
    dataset_config_all = json.load(open(path, "r"))
    name = dataset + "_" + version
    if name in dataset_config_all:
        return DatasetConfig.from_json(dataset_config_all[name])
    else:
        return None


def load_dataset_label_names(dataset_config, label_index):
    for p in dir(dataset_config):
        if getattr(dataset_config, p) == label_index and "label_index" in p:
            temp = p.split("_")
            label_num = getattr(dataset_config, temp[0] + "_" + temp[1] + "_size")
            if hasattr(dataset_config, temp[0] + "_" + temp[1]):
                return getattr(dataset_config, temp[0] + "_" + temp[1]), label_num
            else:
                return None, label_num
    return None, -1
