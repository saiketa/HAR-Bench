import os
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import TrainConfig
from models.LIMU_BERT import LIMUBertModel4Pretrain
from models.TS_TCC import TSTCC4Pretrain
from models.TS2Vec import TS2VecModel4Pretrain
from models.SimMTM import SimMTMModel4Pretrain
from models.BioBankSSL import BioBankSSL4Pretrain
from models.CrossHAR import MaskedModel4Pretrain as CrossHARMaskedModel4Pretrain
from models.CRT import CRT4Pretrain
from models.FOCAL import FOCAL4Pretrain, adapt_focal_checkpoint
from utils.utils import (
    handle_embedding_argv,
    get_device,
    set_seeds,
    load_multiple_pretrain_datasets,
    slice_imu_channels,
    update_model_input_config,
)
from utils.preprocessors import IMUDataset, Preprocess4CRT


def build_model_for_embedding(mode, model_cfg, output_embed=True):
    """
    Build model for embedding extraction.

    Returns:
        model
    """
    if mode == "LIMU-BERT":
        model = LIMUBertModel4Pretrain(model_cfg, output_embed=output_embed)

    elif mode == "TS-TCC":
        # returns (predictions, features)
        model = TSTCC4Pretrain(model_cfg)

    elif mode == "TS2Vec":
        model = TS2VecModel4Pretrain(model_cfg)

    elif mode == "SimMTM":
        model = SimMTMModel4Pretrain(model_cfg)

    elif mode == "BioBankSSL":
        # returns (aot_pred, permute_pred, time_w_pred, features)
        model = BioBankSSL4Pretrain(model_cfg)

    elif mode == "CrossHAR":
        # returns [B, T, H] when output_embed=True
        model = CrossHARMaskedModel4Pretrain(model_cfg, output_embed=output_embed)

    elif mode == "CRT":
        model = CRT4Pretrain(model_cfg)

    elif mode == "FOCAL":
        model = FOCAL4Pretrain(model_cfg)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return model


def fetch_setup_for_one_dataset(args, data, labels, output_embed=True):
    train_cfg = TrainConfig.from_json(args.train_cfg)
    model_cfg = update_model_input_config(args.model_cfg, args.input_channels)

    set_seeds(train_cfg.seed)

    feature_len = model_cfg.feature_num
    data = slice_imu_channels(data, feature_len)

    pipeline = []
    cfg_for_embed = model_cfg
    if args.mode == "CRT":
        # Keep embedding data path aligned with CRT pretraining:
        # instance-norm -> time/freq preprocess -> [2T, C].
        pipeline = [Preprocess4CRT(feature_len=feature_len, return_tensor=False)]

    data_set = IMUDataset(
        data,
        labels,
        feature_len=model_cfg.feature_num,
        pipeline=pipeline,
        isInstanceNorm=True,
    )
    data_loader = DataLoader(
        data_set,
        shuffle=False,
        batch_size=train_cfg.batch_size
    )

    if args.mode == "CRT":
        sample_x, _ = data_set[0]  # [2T, C]
        sample_len = int(sample_x.shape[0])
        if hasattr(model_cfg, "_replace"):
            cfg_for_embed = model_cfg._replace(seq_len=sample_len)
        else:
            cfg_for_embed = SimpleNamespace(**vars(model_cfg))
            cfg_for_embed.seq_len = sample_len

        if cfg_for_embed.seq_len % (4 * cfg_for_embed.patch_len) != 0:
            raise ValueError(
                f"Invalid CRT config for embedding: seq_len={cfg_for_embed.seq_len}, "
                f"patch_len={cfg_for_embed.patch_len}, require seq_len % (4 * patch_len) == 0."
            )

    model = build_model_for_embedding(
        mode=args.mode,
        model_cfg=cfg_for_embed,
        output_embed=output_embed
    )

    criterion = nn.MSELoss(reduction='none')
    return data, labels, data_loader, model, criterion, train_cfg


def load_checkpoint_for_embedding(
    model,
    ckpt_path,
    device,
    mode,
    lambda1=6.0,
    lambda2=1.0,
):
    """
    Load checkpoint according to mode.

    LIMU-BERT:
        checkpoint is plain state_dict

    TS-TCC / BioBankSSL:
        checkpoint is dict with 'model_state_dict'

    CrossHAR:
        checkpoint is masked-model plain state_dict
        path pattern: xxx_masked_{lambda1}_{lambda2}.pt
    """
    if mode == "CrossHAR":
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
        return

    checkpoint = torch.load(ckpt_path, map_location=device)

    if mode == "LIMU-BERT":
        model.load_state_dict(checkpoint)

    elif mode == "FOCAL":
        checkpoint = adapt_focal_checkpoint(
            checkpoint,
            use_dual_modalities=getattr(model.backbone, "use_dual_modalities", True),
        )
        model.load_state_dict(checkpoint)

    elif mode == "TS2Vec":
        if isinstance(checkpoint, dict) and "_net" in checkpoint and "net" in checkpoint:
            model._net.load_state_dict(checkpoint["_net"])
            model.net.load_state_dict(checkpoint["net"])
        else:
            model.load_state_dict(checkpoint)

    elif mode in ["TS-TCC", "BioBankSSL", "CRT", "SimMTM"]:
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"{mode} checkpoint must contain 'model_state_dict', "
                f"got keys: {list(checkpoint.keys())}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def generate_embedding_for_one_dataset(args, dataset_name, data, labels, output_embed=True):
    data, labels, data_loader, model, criterion, train_cfg = fetch_setup_for_one_dataset(
        args, data, labels, output_embed
    )

    device = get_device(args.gpu)
    model = model.to(device)

    if args.mode == "CrossHAR":
        lambda1 = getattr(args, "lambda1", 6.0)
        lambda2 = getattr(args, "lambda2", 1.0)
        ckpt_file = f"{args.save_path}_masked_{lambda1}_{lambda2}.pt"
    else:
        ckpt_file = args.save_path + ".pt"

    load_checkpoint_for_embedding(model, ckpt_file, device, args.mode)
    model.eval()

    outputs = []

    with torch.no_grad():
        for batch in data_loader:
            seqs, label = batch
            seqs = seqs.to(device).float()

            if args.mode == "LIMU-BERT":
                # [B, T, H]
                embed = model(seqs)

            elif args.mode == "TS-TCC":
                # predictions: [B, num_classes]
                # features: [B, H, T]
                _, features = model(seqs)
                embed = features.transpose(1, 2)   # -> [B, T, H]

            elif args.mode == "TS2Vec":
                embed = model.backbone_features(seqs)  # [B, T, H]

            elif args.mode == "SimMTM":
                embed = model.backbone_features(seqs)  # [B, T, H]

            elif args.mode == "BioBankSSL":
                # aot_pred, permute_pred, time_w_pred, features
                _, _, _, features = model(seqs)
                embed = features.transpose(1, 2)   # -> [B, T, H]

            elif args.mode == "CrossHAR":
                # output_embed=True -> [B, T, H]
                embed = model(seqs)

            elif args.mode == "CRT":
                # [B, D]
                embed = model.backbone_features(seqs)

            elif args.mode == "FOCAL":
                # [B, T, H]
                embed = model.backbone_features(seqs)

            else:
                raise ValueError(f"Unsupported mode: {args.mode}")

            outputs.append(embed.detach().cpu().numpy())

    output = np.concatenate(outputs, axis=0).astype(np.float32)

    print(f"[{dataset_name}] embedding shape: {output.shape}, label shape: {labels.shape}")
    return data, output, labels


def generate_embedding_or_output(
    args,
    save=False,
    output_embed=True,
    lambda1=6.0,
    lambda2=1.0,
):
    train_cfg = TrainConfig.from_json(args.train_cfg)
    set_seeds(train_cfg.seed)

    # for CrossHAR checkpoint naming
    args.lambda1 = lambda1
    args.lambda2 = lambda2

    data_list, label_list = load_multiple_pretrain_datasets(
        datasets_root=args.datasets_root,
        dataset_names=args.pretrain_datasets,
        dataset_version=args.dataset_version
    )

    args.model_cfg = update_model_input_config(args.model_cfg, args.input_channels)

    all_embeddings = []
    all_labels = []
    all_dataset_ids = []
    results = {}

    for dataset_idx, (dataset_name, data, labels) in enumerate(
        zip(args.pretrain_datasets, data_list, label_list)
    ):
        print(f"\n=== Generating embedding for dataset: {dataset_name} ===")
        data, output, labels = generate_embedding_for_one_dataset(
            args=args,
            dataset_name=dataset_name,
            data=data,
            labels=labels,
            output_embed=output_embed
        )

        output = np.asarray(output, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        # unify labels to [N, 1, 1]
        if labels.ndim == 3:
            labels = labels[:, :, 0:1]
        elif labels.ndim == 2:
            labels = labels[:, :, None]
        else:
            raise ValueError(
                f"Unexpected label shape for dataset {dataset_name}: {labels.shape}"
            )

        print(
            f"[{dataset_name}] unified embedding shape: {output.shape}, "
            f"unified label shape: {labels.shape}"
        )

        all_embeddings.append(output)
        all_labels.append(labels)
        all_dataset_ids.append(
            np.full((output.shape[0],), dataset_idx, dtype=np.int64)
        )

        results[dataset_name] = {
            "data": data,
            "embedding": output,
            "labels": labels
        }

    merged_embedding = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    merged_label = np.concatenate(all_labels, axis=0).astype(np.float32)
    merged_dataset_id = np.concatenate(all_dataset_ids, axis=0).astype(np.int64)

    print("\n=== Merged result ===")
    print("merged embedding shape:", merged_embedding.shape)
    print("merged label shape:", merged_label.shape)
    print("merged_dataset_id shape:", merged_dataset_id.shape)

    if save:
        # os.makedirs("embed", exist_ok=True)
        os.makedirs(args.embed_dir, exist_ok=True)

        ckpt_name = os.path.basename(args.save_path)

        if args.mode == "CrossHAR":
            ckpt_name = f"{ckpt_name}_masked_{lambda1}_{lambda2}"

        embed_save_name = f"embedding_{ckpt_name}.npy"
        label_save_name = f"label_{ckpt_name}.npy"
        dataset_id_save_name = f"dataset_id_{ckpt_name}.npy"

        embed_save_path = os.path.join(args.embed_dir, embed_save_name)
        label_save_path = os.path.join(args.embed_dir, label_save_name)
        dataset_id_save_path = os.path.join(args.embed_dir, dataset_id_save_name)

        np.save(embed_save_path, merged_embedding)
        np.save(label_save_path, merged_label)
        np.save(dataset_id_save_path, merged_dataset_id)

        print(f"Saved merged embedding to: {embed_save_path}")
        print(f"Saved merged label to: {label_save_path}")
        print(f"Saved merged dataset_id to: {dataset_id_save_path}")

    return {
        "results_by_dataset": results,
        "merged_embedding": merged_embedding,
        "merged_label": merged_label,
        "merged_dataset_id": merged_dataset_id,
    }


def load_embedding_label(target, embed_dir="embed"):
    embed_name = f"embedding_{target}.npy"
    label_name = f"label_{target}.npy"

    embed = np.load(os.path.join(embed_dir, embed_name)).astype(np.float32)
    labels = np.load(os.path.join(embed_dir, label_name)).astype(np.float32)
    return embed, labels


if __name__ == "__main__":
    save = True
    args = handle_embedding_argv(default_mode="BioBankSSL")
    mode = args.mode

    if mode == "CrossHAR":
        lambda1 = 6.0
        lambda2 = 1.0
        ckpt_file = f"{args.save_path}_masked_{lambda1}_{lambda2}.pt"
    else:
        lambda1 = None
        lambda2 = None
        ckpt_file = args.save_path + ".pt"

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_file}")

    results = generate_embedding_or_output(
        args=args,
        output_embed=True,
        save=save,
        lambda1=6.0 if mode == "CrossHAR" else 6.0,
        lambda2=1.0 if mode == "CrossHAR" else 1.0,
    )
