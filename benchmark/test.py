import os
import argparse
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from config import TrainConfig, load_model_config
from models.LIMU_BERT import LIMUBertModel4Pretrain
from models.TS_TCC import TSTCC4Pretrain
from models.TS2Vec import TS2VecModel4Pretrain
from models.SimMTM import SimMTMModel4Pretrain
from models.BioBankSSL import BioBankSSL4Pretrain
from models.CrossHAR import MaskedModel4Pretrain as CrossHARMaskedModel4Pretrain
from models.CRT import CRT4Pretrain
from models.FOCAL import FOCAL4Pretrain, adapt_focal_checkpoint
from models.classifiers import fetch_classifier

from utils.utils import (
    get_device,
    set_seeds,
    get_imu_input_tag,
    slice_imu_channels,
    update_model_input_config,
)
from utils.preprocessors import IMUDataset, Preprocess4CRT

def parse_args():
    parser = argparse.ArgumentParser(description='Cross-dataset test')

    parser.add_argument('--datasets_root', type=str, default='./datasets',
                        help='Root dir containing dataset folders')
    parser.add_argument('--dataset_version', type=str, default='20_120',
                        choices=['10_100', '20_120'])
    parser.add_argument('-td', '--dataset_names', nargs='+', default=None,
                        help='Specific dataset folder names to test; default: all folders under datasets_root')

    parser.add_argument('--mode', type=str, default='BioBankSSL',
                        choices=['LIMU-BERT', 'TS-TCC', 'TS2Vec', 'SimMTM', 'BioBankSSL', 'FOCAL', 'CrossHAR', 'CRT'],
                        help='Pretraining method used for embedding extraction')
    parser.add_argument('--encoder_backbone', type=str, default='transformer',
                        choices=['transformer', 'cnn', 'resnet'],
                        help='Encoder backbone used by the pretrained checkpoint')

    parser.add_argument('--model_root', type=str, default='./saved',
                        help='Root directory for saved checkpoints')

    parser.add_argument('--pretrain_train_cfg', type=str, default=None)
    parser.add_argument('--classifier_train_cfg', type=str, default='./config/classifier.json')

    parser.add_argument('--model_version', type=str, default='v1')

    parser.add_argument('--classifier_prefix', type=str, default='transformer',
                        help='Classifier type, e.g. gru / cnn / mlp / transformer')

    parser.add_argument('--lambda1', type=float, default=6.0,
                        help='CrossHAR loss weight lambda1')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='CrossHAR loss weight lambda2')

    parser.add_argument('-g', '--gpu', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for testing')
    parser.add_argument('--output_csv', type=str, default='./cross_dataset_results.csv')
    parser.add_argument('--input_channels', type=int, default=6, choices=[3, 6],
                        help='Use 3-axis accelerometer only or full 6-axis IMU input')

    return parser.parse_args()


def resolve_model_names(args):
    """
    Resolve model naming convention from mode and classifier_prefix.

    Returns:
        pretrain_target, pretrain_prefix, classifier_target
    """
    mode_to_prefix = {
        'LIMU-BERT': 'LIMU-BERT',
        'TS-TCC': 'TS-TCC',
        'TS2Vec': 'TS2Vec',
        'SimMTM': 'SimMTM',
        'BioBankSSL': 'BioBankSSL',
        'FOCAL': 'FOCAL',
        'CrossHAR': 'CrossHAR',
        'CRT': 'CRT',
    }

    if args.mode not in mode_to_prefix:
        raise ValueError(f'Unsupported mode: {args.mode}')

    pretrain_prefix = mode_to_prefix[args.mode]
    pretrain_target = f'pretrain_{pretrain_prefix}'
    classifier_target = f'classifier_{pretrain_prefix}_{args.classifier_prefix}'

    return pretrain_target, pretrain_prefix, classifier_target


def resolve_checkpoint_paths(args):
    """
    Build checkpoint paths automatically from mode and classifier_prefix.

    Returns:
        pretrain_ckpt_file, classifier_ckpt_file
    """
    _, pretrain_prefix, _ = resolve_model_names(args)
    input_tag = get_imu_input_tag(args.input_channels)

    if args.mode == 'CrossHAR':
        pretrain_ckpt_file = os.path.join(
            args.model_root,
            f'pretrain_{pretrain_prefix}_{input_tag}_masked_{args.lambda1}_{args.lambda2}.pt'
        )
    else:
        pretrain_ckpt_file = os.path.join(
            args.model_root,
            f'pretrain_{pretrain_prefix}_{input_tag}.pt'
        )

    classifier_ckpt_file = os.path.join(
        args.model_root,
        f'classifier_{pretrain_prefix}_{args.classifier_prefix}_{input_tag}.pt'
    )

    return pretrain_ckpt_file, classifier_ckpt_file


def get_dataset_names(datasets_root, dataset_names=None, dataset_version='20_120'):
    if dataset_names is not None and len(dataset_names) > 0:
        return dataset_names

    names = []
    for x in sorted(os.listdir(datasets_root)):
        full = os.path.join(datasets_root, x)
        if os.path.isdir(full):
            version_data_file = os.path.join(full, f'data_{dataset_version}.npy')
            version_label_file = os.path.join(full, f'label_{dataset_version}.npy')
            raw_data_file = os.path.join(full, 'data.npy')
            raw_label_file = os.path.join(full, 'label.npy')
            if (
                os.path.exists(version_data_file) and os.path.exists(version_label_file)
            ) or (
                os.path.exists(raw_data_file) and os.path.exists(raw_label_file)
            ):
                names.append(x)
    return names


def load_one_dataset(datasets_root, dataset_name, dataset_version, keep_all_label_dims=False):
    dataset_dir = os.path.join(datasets_root, dataset_name)
    data_path = os.path.join(dataset_dir, f'data_{dataset_version}.npy')
    label_path = os.path.join(dataset_dir, f'label_{dataset_version}.npy')

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        data_path = os.path.join(dataset_dir, 'data.npy')
        label_path = os.path.join(dataset_dir, 'label.npy')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data file not found: {data_path}')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f'Label file not found: {label_path}')

    data = np.load(data_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)

    if labels.ndim == 2:
        labels = labels[:, :, None]
    elif labels.ndim != 3:
        raise ValueError(f'Unexpected label shape for dataset {dataset_name}: {labels.shape}')

    if not keep_all_label_dims:
        labels = labels[:, :, 0:1]

    return data, labels


def infer_global_label_space(datasets_root, dataset_names, dataset_version):
    """Infer classifier output dim from the union of activity ids across datasets."""
    all_activity_ids = []
    for dataset_name in dataset_names:
        _, labels = load_one_dataset(datasets_root, dataset_name, dataset_version)
        activity_ids = np.unique(labels[:, 0, 0].astype(np.int64))
        all_activity_ids.append(activity_ids)

    merged_ids = np.unique(np.concatenate(all_activity_ids, axis=0))
    label_num = int(merged_ids.max()) + 1
    return label_num, merged_ids


def build_pretrain_model(args):
    pretrain_target, pretrain_prefix, _ = resolve_model_names(args)

    model_cfg = load_model_config(
        pretrain_target,
        pretrain_prefix,
        args.model_version,
        encoder_backbone=getattr(args, 'encoder_backbone', 'transformer'),
    )
    if model_cfg is None:
        raise ValueError(
            f'Unable to load pretrain model config: target={pretrain_target}, prefix={pretrain_prefix}'
        )
    model_cfg = update_model_input_config(model_cfg, args.input_channels)

    if args.mode == 'LIMU-BERT':
        model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)

    elif args.mode == 'TS-TCC':
        model = TSTCC4Pretrain(model_cfg)
    elif args.mode == 'TS2Vec':
        model = TS2VecModel4Pretrain(model_cfg)
    elif args.mode == 'SimMTM':
        model = SimMTMModel4Pretrain(model_cfg)

    elif args.mode == 'BioBankSSL':
        model = BioBankSSL4Pretrain(model_cfg)

    elif args.mode == 'CrossHAR':
        model = CrossHARMaskedModel4Pretrain(model_cfg, output_embed=True)
    elif args.mode == 'FOCAL':
        model = FOCAL4Pretrain(model_cfg)
    elif args.mode == 'CRT':
        model = CRT4Pretrain(model_cfg)

    else:
        raise ValueError(f'Unsupported pretrain mode: {args.mode}')

    return model_cfg, model


def build_classifier_model(args, input_dim, output_dim, seq_len=None):
    _, _, classifier_target = resolve_model_names(args)

    model_cfg = load_model_config(classifier_target, args.classifier_prefix, args.model_version)
    if model_cfg is None:
        raise ValueError(
            f'Unable to load classifier model config: target={classifier_target}, prefix={args.classifier_prefix}'
        )

    if seq_len is not None:
        if hasattr(model_cfg, "_replace"):
            model_cfg = model_cfg._replace(seq_len=int(seq_len))
        else:
            model_cfg = SimpleNamespace(**vars(model_cfg))
            model_cfg.seq_len = int(seq_len)

    model = fetch_classifier(args.classifier_prefix, model_cfg, input=input_dim, output=output_dim)
    if model is None:
        raise ValueError(f'Unsupported classifier type: {args.classifier_prefix}')

    return model_cfg, model


def load_pretrain_checkpoint(model, ckpt_file, device, mode):
    checkpoint = torch.load(ckpt_file, map_location=device)

    if mode == 'LIMU-BERT':
        model.load_state_dict(checkpoint)

    elif mode == 'FOCAL':
        checkpoint = adapt_focal_checkpoint(
            checkpoint,
            use_dual_modalities=getattr(model.backbone, "use_dual_modalities", True),
        )
        model.load_state_dict(checkpoint)

    elif mode == 'TS2Vec':
        if isinstance(checkpoint, dict) and '_net' in checkpoint and 'net' in checkpoint:
            model._net.load_state_dict(checkpoint['_net'])
            model.net.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)

    elif mode in ['TS-TCC', 'BioBankSSL', 'CRT', 'SimMTM']:
        if 'model_state_dict' not in checkpoint:
            raise KeyError(
                f'{mode} checkpoint must contain "model_state_dict", got keys: {list(checkpoint.keys())}'
            )
        model.load_state_dict(checkpoint['model_state_dict'])

    elif mode == 'CrossHAR':
        model.load_state_dict(checkpoint)

    else:
        raise ValueError(f'Unsupported pretrain mode: {mode}')


def load_classifier_checkpoint(model, ckpt_file, device):
    checkpoint = torch.load(ckpt_file, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)


def forward_for_embedding(pretrain_model, seqs, mode, labels=None, dataset_name=None):
    """
    Return embedding unified as [B, T, H].
    """
    if mode == 'LIMU-BERT':
        embed = pretrain_model(seqs)   # [B, T, H]

    elif mode == 'TS-TCC':
        _, features = pretrain_model(seqs)   # [B, H, T]
        embed = features.transpose(1, 2)     # [B, T, H]
    elif mode == 'TS2Vec':
        embed = pretrain_model.backbone_features(seqs)  # [B, T, H]
    elif mode == 'SimMTM':
        embed = pretrain_model.backbone_features(seqs)  # [B, T, H]

    elif mode == 'BioBankSSL':
        _, _, _, features = pretrain_model(seqs)  # [B, H, T]
        embed = features.transpose(1, 2)          # [B, T, H]

    elif mode == 'CrossHAR':
        embed = pretrain_model(seqs)   # [B, T, H]
    elif mode == 'FOCAL':
        embed = pretrain_model.backbone_features(seqs)  # [B, T, H]
    elif mode == 'CRT':
        embed = pretrain_model.backbone_features(seqs)  # [B, D]

    else:
        raise ValueError(f'Unsupported pretrain mode: {mode}')

    return embed


def extract_embeddings(pretrain_model, data, labels, feature_len, batch_size, device, mode, dataset_name=None, seq_len=None):
    data = slice_imu_channels(data, feature_len)
    pipeline = []
    if mode == 'CRT':
        pipeline = [Preprocess4CRT(feature_len=feature_len, return_tensor=False)]
    dataset = IMUDataset(
        data,
        labels,
        feature_len=feature_len,
        pipeline=pipeline,
        isInstanceNorm=True
    )
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    all_embeddings = []
    all_labels = []

    pretrain_model.eval()
    with torch.no_grad():
        for seqs, label in loader:
            seqs = seqs.to(device).float()
            label = label.to(device)
            embed = forward_for_embedding(pretrain_model, seqs, mode, labels=label, dataset_name=dataset_name)
            all_embeddings.append(embed.detach().cpu())
            all_labels.append(label.detach().cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)

    if labels.ndim == 3:
        labels = labels[:, 0, 0]
    elif labels.ndim == 2:
        labels = labels[:, 0]
    elif labels.ndim == 1:
        pass
    else:
        raise ValueError(f'Unexpected extracted label shape: {labels.shape}')

    return embeddings, labels


def evaluate_classifier(classifier_model, embeddings, labels, batch_size, device):
    if embeddings.ndim == 2:
        embeddings = embeddings[:, None, :]
    elif embeddings.ndim != 3:
        raise ValueError(f'Unexpected embedding shape: {embeddings.shape}')

    dataset = IMUDataset(
        embeddings.astype(np.float32),
        labels.astype(np.int64),
        isInstanceNorm=False
    )
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    all_logits = []
    all_labels = []

    classifier_model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            logits = classifier_model(x, False)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_pred = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return acc, f1, y_true, y_pred


# def main():
#     args = parse_args()

#     pretrain_train_cfg = TrainConfig.from_json(args.pretrain_train_cfg)
#     classifier_train_cfg = TrainConfig.from_json(args.classifier_train_cfg)

#     seed = getattr(classifier_train_cfg, 'seed', 42)
#     set_seeds(seed)

#     device = get_device(args.gpu)
#     test_batch_size = args.batch_size if args.batch_size is not None else classifier_train_cfg.batch_size

#     dataset_names = get_dataset_names(
#         args.datasets_root,
#         args.dataset_names,
#         dataset_version=args.dataset_version
#     )
#     if len(dataset_names) == 0:
#         raise ValueError(f'No valid datasets found under {args.datasets_root}')

#     pretrain_target, pretrain_prefix, classifier_target = resolve_model_names(args)
#     pretrain_ckpt_file, classifier_ckpt_file = resolve_checkpoint_paths(args)

#     print('Mode:', args.mode)
#     print('Datasets to test:', dataset_names)
#     print('Pretrain target:', pretrain_target)
#     print('Pretrain prefix:', pretrain_prefix)
#     print('Classifier target:', classifier_target)
#     print('Classifier prefix:', args.classifier_prefix)
#     print('Pretrain ckpt:', pretrain_ckpt_file)
#     print('Classifier ckpt:', classifier_ckpt_file)

#     pretrain_model_cfg, pretrain_model = build_pretrain_model(args)
#     if not os.path.exists(pretrain_ckpt_file):
#         raise FileNotFoundError(f'Pretrain checkpoint not found: {pretrain_ckpt_file}')

#     load_pretrain_checkpoint(
#         model=pretrain_model,
#         ckpt_file=pretrain_ckpt_file,
#         device=device,
#         mode=args.mode
#     )
#     pretrain_model = pretrain_model.to(device)
#     print(f'Loaded pretrain model from {pretrain_ckpt_file}')

#     results = []

#     # infer classifier output dim from first dataset
#     first_data, first_labels = load_one_dataset(args.datasets_root, dataset_names[0], args.dataset_version)
#     first_labels_flat = first_labels[:, 0, 0]
#     label_num = len(np.unique(first_labels_flat))

#     warm_embeddings, _ = extract_embeddings(
#         pretrain_model=pretrain_model,
#         data=first_data,
#         labels=first_labels,
#         feature_len=pretrain_model_cfg.feature_num,
#         batch_size=test_batch_size,
#         device=device,
#         mode=args.mode
#     )
#     input_dim = warm_embeddings.shape[-1]

#     classifier_model_cfg, classifier_model = build_classifier_model(
#         args,
#         input_dim=input_dim,
#         output_dim=label_num
#     )

#     if not os.path.exists(classifier_ckpt_file):
#         raise FileNotFoundError(f'Classifier checkpoint not found: {classifier_ckpt_file}')

#     load_classifier_checkpoint(classifier_model, classifier_ckpt_file, device)
#     classifier_model = classifier_model.to(device)
#     print(f'Loaded classifier model from {classifier_ckpt_file}')

#     for dataset_name in dataset_names:
#         print(f'\n=== Testing on dataset: {dataset_name} ===')
#         data, labels = load_one_dataset(args.datasets_root, dataset_name, args.dataset_version)
#         print(f'data shape: {data.shape}, label shape: {labels.shape}')

#         embeddings, labels_flat = extract_embeddings(
#             pretrain_model=pretrain_model,
#             data=data,
#             labels=labels,
#             feature_len=pretrain_model_cfg.feature_num,
#             batch_size=test_batch_size,
#             device=device,
#             mode=args.mode
#         )
#         print(f'embedding shape: {embeddings.shape}, labels shape: {labels_flat.shape}')

#         dataset_label_num = len(np.unique(labels_flat))
#         if dataset_label_num != label_num:
#             print(
#                 f'Warning: label count in {dataset_name} is {dataset_label_num}, '
#                 f'while classifier output dim is {label_num}'
#             )

#         acc, f1, _, _ = evaluate_classifier(
#             classifier_model=classifier_model,
#             embeddings=embeddings,
#             labels=labels_flat,
#             batch_size=test_batch_size,
#             device=device
#         )

#         print(f'[{dataset_name}] Acc: {acc:.4f}, Macro-F1: {f1:.4f}')

#         results.append({
#             'dataset': dataset_name,
#             'num_samples': len(labels_flat),
#             'acc': acc,
#             'macro_f1': f1,
#         })

#     df = pd.DataFrame(results)
#     print('\n=== Summary ===')
#     print(df)

#     if args.output_csv:
#         df.to_csv(args.output_csv, index=False)
#         print(f'Saved results to {args.output_csv}')

def main():
    args = parse_args()
    if args.pretrain_train_cfg is None:
        args.pretrain_train_cfg = f'./config/{args.mode}.json'

    pretrain_train_cfg = TrainConfig.from_json(args.pretrain_train_cfg)
    classifier_train_cfg = TrainConfig.from_json(args.classifier_train_cfg)

    seed = getattr(classifier_train_cfg, 'seed', 42)
    set_seeds(seed)

    device = get_device(args.gpu)
    test_batch_size = args.batch_size if args.batch_size is not None else classifier_train_cfg.batch_size

    dataset_names = get_dataset_names(
        args.datasets_root,
        args.dataset_names,
        dataset_version=args.dataset_version
    )
    if len(dataset_names) == 0:
        raise ValueError(f'No valid datasets found under {args.datasets_root}')

    pretrain_target, pretrain_prefix, classifier_target = resolve_model_names(args)
    pretrain_ckpt_file, classifier_ckpt_file = resolve_checkpoint_paths(args)

    print('Mode:', args.mode)
    print('Datasets to test:', dataset_names)
    print('Pretrain target:', pretrain_target)
    print('Pretrain prefix:', pretrain_prefix)
    print('Classifier target:', classifier_target)
    print('Classifier prefix:', args.classifier_prefix)
    print('Pretrain ckpt:', pretrain_ckpt_file)
    print('Classifier ckpt:', classifier_ckpt_file)

    pretrain_model_cfg, pretrain_model = build_pretrain_model(args)

    # For CRT, seq_len must match post-preprocess length [2T, C].
    if args.mode == 'CRT':
        warm_data, warm_labels = load_one_dataset(
            args.datasets_root,
            dataset_names[0],
            args.dataset_version,
            keep_all_label_dims=False,
        )
        warm_data = warm_data[:, :, :pretrain_model_cfg.feature_num]
        warm_data = slice_imu_channels(warm_data, pretrain_model_cfg.feature_num)
        warm_dataset = IMUDataset(
            warm_data,
            warm_labels,
            feature_len=pretrain_model_cfg.feature_num,
            pipeline=[Preprocess4CRT(feature_len=pretrain_model_cfg.feature_num, return_tensor=False)]
        )
        sample_len = int(warm_dataset[0][0].shape[0])
        if hasattr(pretrain_model_cfg, "_replace"):
            pretrain_model_cfg = pretrain_model_cfg._replace(seq_len=sample_len)
        else:
            pretrain_model_cfg = SimpleNamespace(**vars(pretrain_model_cfg))
            pretrain_model_cfg.seq_len = sample_len
        pretrain_model = CRT4Pretrain(pretrain_model_cfg)
    if not os.path.exists(pretrain_ckpt_file):
        raise FileNotFoundError(f'Pretrain checkpoint not found: {pretrain_ckpt_file}')

    load_pretrain_checkpoint(
        model=pretrain_model,
        ckpt_file=pretrain_ckpt_file,
        device=device,
        mode=args.mode
    )
    pretrain_model = pretrain_model.to(device)
    print(f'Loaded pretrain model from {pretrain_ckpt_file}')

    results = []
    all_y_true = []
    all_y_pred = []
    all_dataset_tags = []

    # Infer classifier output dim from the global activity-id space.
    label_num, global_activity_ids = infer_global_label_space(
        args.datasets_root,
        dataset_names,
        args.dataset_version,
    )

    first_data, first_labels = load_one_dataset(
        args.datasets_root,
        dataset_names[0],
        args.dataset_version,
        keep_all_label_dims=False,
    )

    warm_embeddings, _ = extract_embeddings(
        pretrain_model=pretrain_model,
        data=first_data,
        labels=first_labels,
        feature_len=pretrain_model_cfg.feature_num,
        batch_size=test_batch_size,
        device=device,
        mode=args.mode,
        dataset_name=dataset_names[0],
        seq_len=getattr(pretrain_model_cfg, "seq_len", first_data.shape[1]),
    )
    input_dim = warm_embeddings.shape[-1]
    warm_seq_len = 1 if warm_embeddings.ndim == 2 else warm_embeddings.shape[1]

    classifier_model_cfg, classifier_model = build_classifier_model(
        args,
        input_dim=input_dim,
        output_dim=label_num,
        seq_len=warm_seq_len,
    )

    if not os.path.exists(classifier_ckpt_file):
        raise FileNotFoundError(f'Classifier checkpoint not found: {classifier_ckpt_file}')

    load_classifier_checkpoint(classifier_model, classifier_ckpt_file, device)
    classifier_model = classifier_model.to(device)
    print(f'Loaded classifier model from {classifier_ckpt_file}')
    print(f'Global activity ids: {global_activity_ids.tolist()}')
    print(f'Classifier output dim: {label_num}')

    for dataset_name in dataset_names:
        print(f'\n=== Testing on dataset: {dataset_name} ===')
        data, labels = load_one_dataset(
            args.datasets_root,
            dataset_name,
            args.dataset_version,
            keep_all_label_dims=False,
        )
        print(f'data shape: {data.shape}, label shape: {labels.shape}')

        embeddings, labels_flat = extract_embeddings(
            pretrain_model=pretrain_model,
            data=data,
            labels=labels,
            feature_len=pretrain_model_cfg.feature_num,
            batch_size=test_batch_size,
            device=device,
            mode=args.mode,
            dataset_name=dataset_name,
            seq_len=getattr(pretrain_model_cfg, "seq_len", data.shape[1]),
        )
        print(f'embedding shape: {embeddings.shape}, labels shape: {labels_flat.shape}')

        invalid_labels = np.setdiff1d(np.unique(labels_flat), np.arange(label_num))
        if invalid_labels.size > 0:
            raise ValueError(
                f'Dataset {dataset_name} contains labels outside classifier range [0, {label_num - 1}]: '
                f'{invalid_labels.tolist()}'
            )

        dataset_label_num = len(np.unique(labels_flat))
        if dataset_label_num != label_num:
            print(
                f'Warning: label count in {dataset_name} is {dataset_label_num}, '
                f'while classifier output dim is {label_num}'
            )

        acc, f1, y_true, y_pred = evaluate_classifier(
            classifier_model=classifier_model,
            embeddings=embeddings,
            labels=labels_flat,
            batch_size=test_batch_size,
            device=device
        )

        print(f'[{dataset_name}] Acc: {acc:.4f}, Macro-F1: {f1:.4f}')

        results.append({
            'dataset': dataset_name,
            'num_samples': len(labels_flat),
            'acc': acc,
            'macro_f1': f1,
        })

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_dataset_tags.extend([dataset_name] * len(y_true))

    # overall metrics across all datasets
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_macro_f1 = f1_score(all_y_true, all_y_pred, average='macro')

    print('\n=== Overall metrics across ALL datasets ===')
    print(f'Overall Acc: {overall_acc:.4f}')
    print(f'Overall Macro-F1: {overall_macro_f1:.4f}')
    print(f'Total samples: {len(all_y_true)}')

    df = pd.DataFrame(results)
    overall_row = pd.DataFrame([{
        'dataset': 'ALL_DATASETS',
        'num_samples': len(all_y_true),
        'acc': overall_acc,
        'macro_f1': overall_macro_f1,
    }])
    df = pd.concat([df, overall_row], ignore_index=True)

    print('\n=== Summary ===')
    print(df)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f'Saved results to {args.output_csv}')

if __name__ == '__main__':
    main()
