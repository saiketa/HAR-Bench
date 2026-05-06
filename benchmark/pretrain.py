# pretrain.py
import math
import os
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import trainers.trainer_BioBankSSL as trainer_BioBankSSL
import trainers.trainer_LIMU_BERT as trainer_LIMU_BERT
import trainers.trainer_TS_TCC as trainer_TS_TCC
import trainers.trainer_CrossHAR as trainer_CrossHAR
import trainers.trainer_CRT as trainer_CRT
import trainers.trainer_FOCAL as trainer_FOCAL
import trainers.trainer_TS2Vec as trainer_TS2Vec
import trainers.trainer_SimMTM as trainer_SimMTM

from models.LIMU_BERT import LIMUBertModel4Pretrain
from models.TS_TCC import TSTCC4Pretrain, TC
from models.BioBankSSL import BioBankSSL4Pretrain
from models.CrossHAR import MaskedModel4Pretrain, Contrastive
from models.CRT import CRT4Pretrain
from models.FOCAL import FOCAL4Pretrain, FOCALLoss
from models.TS2Vec import TS2VecModel4Pretrain
from models.SimMTM import SimMTMModel4Pretrain

from utils.utils import (
    get_device,
    set_seeds,
    load_multi_pretrain_data_config,
    load_multiple_pretrain_datasets,
    prepare_multi_pretrain_dataset,
    subsample_training_subset,
    handle_pretrain_argv,
)
from utils.preprocessors import (
    Preprocess4Mask,
    Preprocess4Aug,
    Preprocess4MultiTask,
    Preprocess4CrossHAR,
    Preprocess4CRT,
    Preprocess4FOCAL,
)
from utils.load_datasets import (
    Dataset4Pretrain,
    Dataset4Contrast,
    Dataset4MultiTask,
    Dataset4CrossHAR,
    Dataset4CRT,
    Dataset4FOCAL,
    Dataset4TS2Vec,
    Dataset4SimMTM,
)


def main_from_data(args, training_rate, mode, data_list, label_list, train_cfg, model_cfg, mask_cfg):
    set_seeds(train_cfg.seed)
    feature_len = model_cfg.feature_num
    seq_len = model_cfg.seq_len
    encoder_name = str(getattr(model_cfg, "encoder_type", "transformer"))

    data_train, label_train, data_vali, label_vali = prepare_multi_pretrain_dataset(
        data_list=data_list,
        label_list=label_list,
        training_rate=training_rate,
        vali_rate=0.2,
        seed=train_cfg.seed,
        shuffle_samples=(mode != "FOCAL"),
    )
    data_train, label_train = subsample_training_subset(
        data_train,
        label_train,
        subset_rate=args.train_subset_rate,
        seed=train_cfg.seed,
        balance=False,
    )
    print(f"Using {args.train_subset_rate:.3f} of the 80% pretrain training split.")

    if args.data_other_subset_rate > 0.0:
        other_data_list, other_label_list = load_multiple_pretrain_datasets(
            datasets_root=args.datasets_other_root,
            dataset_names=args.pretrain_datasets,
            dataset_version=args.dataset_version,
            required=False,
        )
        if other_data_list:
            other_data_list = [
                data[:, :, :feature_len].astype(data_train.dtype, copy=False)
                for data in other_data_list
            ]
            data_other_train, label_other_train, _, _ = prepare_multi_pretrain_dataset(
                data_list=other_data_list,
                label_list=other_label_list,
                training_rate=args.data_other_subset_rate,
                vali_rate=0.0,
                seed=train_cfg.seed,
                shuffle_samples=(mode != "FOCAL"),
            )
            if data_other_train.shape[1:] != data_train.shape[1:]:
                raise ValueError(
                    f"data_other shape mismatch: got {data_other_train.shape[1:]}, "
                    f"expected {data_train.shape[1:]}"
                )
            if label_other_train.shape[1:] != label_train.shape[1:]:
                raise ValueError(
                    f"data_other label shape mismatch: got {label_other_train.shape[1:]}, "
                    f"expected {label_train.shape[1:]}"
                )
            data_train = np.concatenate([data_train, data_other_train], axis=0)
            label_train = np.concatenate([label_train, label_other_train], axis=0)
            if mode != "FOCAL":
                train_idx = np.arange(data_train.shape[0])
                np.random.shuffle(train_idx)
                data_train = data_train[train_idx]
                label_train = label_train[train_idx]
            print(
                f"Added {args.data_other_subset_rate:.3f} of data_other to encoder pretraining: "
                f"{data_other_train.shape[0]} samples."
            )
        else:
            print(
                f"data_other_subset_rate={args.data_other_subset_rate:.3f}, "
                "but no matching data_other datasets were found."
            )

    device = get_device(args.gpu)
    print(f"Encoder backbone: {encoder_name}")

    if mode == "LIMU-BERT":
        pipeline = [Preprocess4Mask(mask_cfg)]

        data_set_train = Dataset4Pretrain(data_train, feature_len=feature_len, pipeline=pipeline)
        data_set_vali = Dataset4Pretrain(data_vali, feature_len=feature_len, pipeline=pipeline)

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)

        model = LIMUBertModel4Pretrain(model_cfg)
        criterion = nn.MSELoss(reduction="none")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)

        trainer = trainer_LIMU_BERT.Trainer(train_cfg, model, optimizer, args.save_path, device)

        def func_loss(model, batch):
            mask_seqs, masked_pos, seqs = batch
            seq_recon = model(mask_seqs, masked_pos)
            return criterion(seq_recon, seqs)

        def func_forward(model, batch):
            mask_seqs, masked_pos, seqs = batch
            seq_recon = model(mask_seqs, masked_pos)
            return seq_recon, seqs

        def func_evaluate(seqs, predict_seqs):
            loss_lm = criterion(predict_seqs, seqs)
            return loss_lm.mean().detach().cpu().numpy()

        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_vali)

    elif mode == "TS-TCC":
        pipeline = [Preprocess4Aug(feature_len=feature_len)]

        data_set_train = Dataset4Contrast(data_train, feature_len=feature_len, pipeline=pipeline)
        data_set_vali = Dataset4Contrast(data_vali, feature_len=feature_len, pipeline=pipeline)

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)

        model = TSTCC4Pretrain(model_cfg).to(device)
        temporal_contr_model = TC(
            bb_dim=model_cfg.hidden,
            device=device,
            tc_hidden=100,
            timestep=6,
            temp_unit="tsfm",
        ).to(device)

        model_optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
        temp_cont_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=train_cfg.lr)

        trainer = trainer_TS_TCC.Trainer(
            cfg=train_cfg,
            model=model,
            temporal_contr_model=temporal_contr_model,
            model_optimizer=model_optimizer,
            temporal_optimizer=temp_cont_optimizer,
            save_path=args.save_path,
            device=device,
        )

        trainer.pretrain(
            data_loader_train=data_loader_train,
            data_loader_vali=data_loader_vali,
            lambda1=1.0,
            lambda2=0.7,
        )

    elif mode == "TS2Vec":
        data_set_train = Dataset4TS2Vec(data_train, feature_len=feature_len, pipeline=[])
        data_set_vali = Dataset4TS2Vec(data_vali, feature_len=feature_len, pipeline=[])

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)

        model = TS2VecModel4Pretrain(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model._net.parameters(), lr=train_cfg.lr)

        trainer = trainer_TS2Vec.Trainer(
            cfg=train_cfg,
            model=model,
            optimizer=optimizer,
            save_path=args.save_path,
            device=device,
        )
        trainer.pretrain(
            data_loader_train=data_loader_train,
            data_loader_vali=data_loader_vali,
        )

    elif mode == "SimMTM":
        data_set_train = Dataset4SimMTM(data_train, feature_len=feature_len, pipeline=[])
        data_set_vali = Dataset4SimMTM(data_vali, feature_len=feature_len, pipeline=[])

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)

        masking_ratio = float(getattr(train_cfg, "simmtm_masking_ratio", 0.5))
        positive_nums = int(getattr(train_cfg, "simmtm_positive_nums", 3))
        if positive_nums <= 0:
            positive_nums = int(math.ceil(1.5 / (1 - masking_ratio)))
        model = SimMTMModel4Pretrain(
            model_cfg,
            temperature=float(getattr(train_cfg, "simmtm_temperature", 0.2)),
            positive_nums=positive_nums,
        ).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)

        trainer = trainer_SimMTM.Trainer(
            cfg=train_cfg,
            model=model,
            optimizer=optimizer,
            save_path=args.save_path,
            device=device,
        )
        trainer.pretrain(
            data_loader_train=data_loader_train,
            data_loader_vali=data_loader_vali,
        )

    elif mode == "FOCAL":
        focal_seq_len = int(getattr(train_cfg, "focal_seq_len", 4))

        data_set_train = Dataset4FOCAL(
            data_train,
            feature_len=feature_len,
            seq_len=focal_seq_len,
            pipeline=[],
        )
        data_set_vali = Dataset4FOCAL(
            data_vali,
            feature_len=feature_len,
            seq_len=focal_seq_len,
            pipeline=[],
        )

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)

        preprocess = Preprocess4FOCAL(
            feature_len=feature_len,
            scaling_sigma=getattr(train_cfg, "focal_scaling_sigma", 0.2),
            time_warp_sigma=getattr(train_cfg, "focal_time_warp_sigma", 0.2),
            time_warp_knots=getattr(train_cfg, "focal_time_warp_knots", 6),
            mag_warp_sigma=getattr(train_cfg, "focal_mag_warp_sigma", 0.05),
            mag_warp_knots=getattr(train_cfg, "focal_mag_warp_knots", 4),
            time_aug_prob=getattr(train_cfg, "focal_time_aug_prob", 0.5),
            phase_shift_prob=getattr(train_cfg, "focal_phase_shift_prob", 0.5),
        )

        model = FOCAL4Pretrain(model_cfg).to(device)
        criterion = FOCALLoss(
            device=device,
            seq_len=focal_seq_len,
            modalities=["acc", "gyro"] if getattr(model_cfg, "focal_use_dual_modalities", True) else ["imu"],
            temperature=getattr(train_cfg, "focal_temperature", 0.07),
            inter_rank_margin=getattr(train_cfg, "focal_inter_rank_margin", 1.0),
            shared_contrastive_loss_weight=getattr(train_cfg, "focal_shared_contrastive_loss_weight", 1.0),
            private_contrastive_loss_weight=getattr(train_cfg, "focal_private_contrastive_loss_weight", 1.0),
            orthogonal_loss_weight=getattr(train_cfg, "focal_orthogonal_loss_weight", 3.0),
            rank_loss_weight=getattr(train_cfg, "focal_rank_loss_weight", 5.0),
            no_private=getattr(train_cfg, "focal_no_private", False),
        ).to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        trainer = trainer_FOCAL.Trainer(
            cfg=train_cfg,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            preprocess=preprocess,
            save_path=args.save_path,
            device=device,
        )
        trainer.pretrain(
            data_loader_train=data_loader_train,
            data_loader_vali=data_loader_vali,
        )

    elif mode == "CRT":
        pipeline = [
            Preprocess4CRT(
                feature_len=feature_len,
                return_tensor=False,
            )
        ]

        data_set_train = Dataset4CRT(data_train, feature_len=feature_len, pipeline=pipeline)
        data_set_vali = Dataset4CRT(data_vali, feature_len=feature_len, pipeline=pipeline)

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)

        # 强制使用预处理后真实长度（2T），并兼容不可变 NamedTuple 配置
        sample_len = int(data_set_train[0].shape[0])
        if hasattr(model_cfg, "_replace"):
            crt_model_cfg = model_cfg._replace(seq_len=sample_len)
        else:
            crt_model_cfg = SimpleNamespace(**vars(model_cfg))
            crt_model_cfg.seq_len = sample_len

        if crt_model_cfg.seq_len % (4 * crt_model_cfg.patch_len) != 0:
            raise ValueError(
                f"Invalid CRT config: seq_len={crt_model_cfg.seq_len}, patch_len={crt_model_cfg.patch_len}, "
                "require seq_len % (4 * patch_len) == 0."
            )

        model = CRT4Pretrain(crt_model_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

        trainer = trainer_CRT.Trainer(
            cfg=train_cfg,
            model=model,
            optimizer=optimizer,
            save_path=args.save_path,
            device=device,
        )

        trainer.pretrain(
            data_loader_train=data_loader_train,
            data_loader_vali=data_loader_vali,
            min_mask_ratio=getattr(train_cfg, "min_mask_ratio", 0.3),
            max_mask_ratio=getattr(train_cfg, "max_mask_ratio", 0.8),
            beta=getattr(train_cfg, "beta", 1e-4),
        )

    elif mode == "BioBankSSL":
        pipeline = [
            Preprocess4MultiTask(
                feature_len=feature_len,
                permute_segments=4,
                time_warp_sigma=0.2,
                p_reverse=0.5,
                p_permute=0.5,
                p_time_warp=0.5,
            )
        ]

        data_set_train = Dataset4MultiTask(data_train, feature_len=feature_len, pipeline=pipeline)
        data_set_vali = Dataset4MultiTask(data_vali, feature_len=feature_len, pipeline=pipeline)

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)

        model = BioBankSSL4Pretrain(model_cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

        trainer = trainer_BioBankSSL.Trainer(
            cfg=train_cfg,
            model=model,
            optimizer=optimizer,
            save_path=args.save_path,
            device=device,
            criterion=criterion,
        )

        trainer.pretrain(
            data_loader_train=data_loader_train,
            data_loader_vali=data_loader_vali,
            lambda_aot=1.0,
            lambda_permute=1.0,
            lambda_time_w=1.0,
        )

    elif mode == "CrossHAR":
        pipeline = [Preprocess4CrossHAR(mask_cfg=mask_cfg)]

        data_set_train = Dataset4CrossHAR(data_train, feature_len=feature_len, pipeline=pipeline)
        data_set_vali = Dataset4CrossHAR(data_vali, feature_len=feature_len, pipeline=pipeline)

        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)

        masked_model = MaskedModel4Pretrain(model_cfg).to(device)
        contrastive_model = Contrastive(model_cfg).to(device)
        criterion = nn.MSELoss(reduction="none")
        masked_optimizer = torch.optim.Adam(params=masked_model.parameters(), lr=train_cfg.lr)
        contrastive_optimizer = torch.optim.Adam(params=contrastive_model.parameters(), lr=train_cfg.lr)

        trainer = trainer_CrossHAR.Trainer(
            cfg=train_cfg,
            masked_model=masked_model,
            masked_optimizer=masked_optimizer,
            contrastive_model=contrastive_model,
            contrastive_optimizer=contrastive_optimizer,
            save_path=args.save_path,
            device=device,
            criterion=criterion,
        )
        trainer.pretrain(
            data_loader_train=data_loader_train,
            data_loader_vali=data_loader_vali,
            lambda1=6.0,
            lambda2=1.0,
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main(args, training_rate, mode):
    data_list, label_list, train_cfg, model_cfg, mask_cfg = load_multi_pretrain_data_config(args)
    main_from_data(args, training_rate, mode, data_list, label_list, train_cfg, model_cfg, mask_cfg)


if __name__ == "__main__":
    args = handle_pretrain_argv(default_mode="LIMU-BERT")
    mode = args.mode
    training_rate = 0.8
    print("Method:", mode)
    print("Config:", args.train_cfg)
    print("Train/validation split: 80% / 20%")
    main(args, training_rate, mode)
