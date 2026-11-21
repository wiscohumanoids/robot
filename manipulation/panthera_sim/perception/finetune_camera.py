"""Fine-tune the deployed Mega Bloks heads on clean and camera-augmented RGB."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adapter import DenseAdapter, normalize, patch_features
from backbone import MODEL_NAME, MODEL_REVISION, load_backbone
from camera_augmentation import CONDITIONS, augment_rgb
from landmarks import LandmarkHead
from train import confusion, scores
from train_landmarks import evaluate as landmark_scores
from train_landmarks import heatmap_loss, point_heatmap


class CameraFrames(Dataset):
    """Vary training exposures by epoch. Keep validation exposures fixed."""

    def __init__(self, root, rows, seed):
        self.root = root
        self.rows = rows
        self.seed = seed
        self.epoch = 1
        self.validation_condition = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = np.random.default_rng([self.seed, self.epoch, row['scene_seed']])
        if self.validation_condition == 'clean':
            condition = 'clean'
        elif self.validation_condition == 'augmented':
            condition = rng.choice(CONDITIONS[1:])
        else:
            condition = 'clean' if rng.random() < .25 else rng.choice(CONDITIONS[1:])
        rgb = augment_rgb(np.array(Image.open(self.root / row['image']).convert('RGB')), condition, rng)
        mask = np.array(Image.open(self.root / row['mask']))
        edges = np.array(Image.open(self.root / row['edges']).convert('L')) > 0
        landmarks = np.stack([point_heatmap(row[key], *mask.shape)
                              for key in ('corners', 'stud_centers')] + [edges]).astype(np.float32)
        return torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(mask), torch.from_numpy(landmarks)


def load_heads(source, device):
    heads = {}
    weights = {}
    for name, filename, key, model_class in (
        ('semantic', 'adapter.pt', 'adapter', DenseAdapter),
        ('landmarks', 'landmarks.pt', 'head', LandmarkHead),
    ):
        checkpoint = torch.load(source / filename, map_location='cpu', weights_only=True)
        if checkpoint['backbone'] != MODEL_NAME or checkpoint['backbone_revision'] != MODEL_REVISION:
            raise ValueError(f'{filename} requires another backbone')
        model = model_class(checkpoint['feature_dim']).to(device)
        model.load_state_dict(checkpoint[key])
        heads[name] = model
        weights[name] = checkpoint
    return heads, weights


def encode(backbone, raw, device):
    rgb = raw.to(device).float() / 255.
    with torch.no_grad():
        features = patch_features(backbone, normalize(rgb))
    return rgb, features


@torch.no_grad()
def cache_validation(dataset, backbone, device, batch_size):
    cached = {}
    for condition in ('clean', 'augmented'):
        dataset.validation_condition = condition
        batches = []
        for raw, target, points in DataLoader(dataset, batch_size=batch_size, num_workers=2):
            _, features = encode(backbone, raw, device)
            batches.append((raw, target, points[:, 2].bool(), features.half().cpu()))
        cached[condition] = batches
    return cached


@torch.no_grad()
def validation(heads, cache, records, thresholds, classes, device):
    for head in heads.values():
        head.eval()
    metrics = {}
    for condition, batches in cache.items():
        matrix = torch.zeros(len(classes), len(classes), dtype=torch.long)
        landmark_logits, edges = [], []
        for raw, mask, edge, features in batches:
            rgb = raw.to(device).float() / 255.
            features = features.to(device).float()
            matrix += confusion(heads['semantic'](features, rgb), mask.to(device).long())
            landmark_logits.append(heads['landmarks'](features, rgb).cpu())
            edges.extend(edge.numpy())
        points = landmark_scores(torch.cat(landmark_logits), records, edges, thresholds)
        metrics[condition] = {
            'semantic': scores(matrix, classes)['foreground_mean_iou'],
            'landmarks': float(np.mean([item['f1'] for item in points.values()])),
            'landmark_f1': {key: item['f1'] for key, item in points.items()},
        }
    return metrics


@torch.no_grad()
def calibrate_landmarks(output, source, model, cache, records, clean_baseline, device):
    """Choose each channel's threshold by paired clean/augmented validation F1."""
    checkpoint = torch.load(output / 'landmarks.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['head'])
    model.eval()
    channels = {'corner': 'corners', 'stud_center': 'stud_centers', 'body_edge': 'edges'}
    originals = checkpoint['thresholds']
    thresholds = sorted(set([.05, .1, .15, .2, .25, .3, .4, .5, .6, .7, *originals.values()]))
    sweeps = {name: {} for name in channels}
    for condition, batches in cache.items():
        logits, edges = [], []
        for raw, _, edge, features in batches:
            rgb = raw.to(device).float() / 255.
            logits.append(model(features.to(device).float(), rgb).cpu())
            edges.extend(edge.numpy())
        logits = torch.cat(logits)
        for threshold in thresholds:
            metrics = landmark_scores(logits, records, edges, dict.fromkeys(channels, threshold))
            for name, key in channels.items():
                sweeps[name].setdefault(threshold, {})[condition] = metrics[key]['f1']
    selected = {}
    for name, key in channels.items():
        feasible = [t for t in thresholds if sweeps[name][t]['clean'] >= clean_baseline[key] - .01]
        if feasible:
            selected[name] = max(feasible, key=lambda t: (
                np.mean(list(sweeps[name][t].values())), -abs(t - originals[name])))
    if len(selected) == len(channels):
        checkpoint['thresholds'] = selected
    else:
        # Deploy the original point and edge head together if a channel
        # cannot retain clean quality at any validation threshold.
        checkpoint = torch.load(source / 'landmarks.pt', weights_only=True, map_location='cpu')
    torch.save(checkpoint, output / 'landmarks.pt')
    return {'sweeps': sweeps, 'selected': checkpoint['thresholds'],
            'retained_source_head': len(selected) < len(channels)}


def semantic_loss(logits, mask, weights):
    ce = F.cross_entropy(logits, mask, weight=weights)
    predicted = logits.softmax(1)
    target = F.one_hot(mask, len(weights)).permute(0, 3, 1, 2)
    dice = (2 * (predicted * target).sum((0, 2, 3)) + 1) / (
        predicted.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1)
    return ce + 1 - dice.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Directory containing both clean checkpoints')
    parser.add_argument('--output', type=Path, required=True, help='New directory for both selected checkpoints')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=718)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.learning_rate) <= 0:
        parser.error('epochs, batch size and learning rate must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    heads, weights = load_heads(args.source, device)
    backbone = load_backbone().to(device)
    metadata = json.loads((args.data / 'landmarks.json').read_text())
    manifest = (args.data / 'manifest.json').read_bytes()
    if metadata['manifest_sha256'] != hashlib.sha256(manifest).hexdigest():
        raise ValueError('Landmark labels and semantic dataset manifest differ')
    frames = metadata['frames']
    train = CameraFrames(args.data, [r for r in frames if r['split'] == 'train'], args.seed)
    val = CameraFrames(args.data, [r for r in frames if r['split'] == 'val'], args.seed + 1)
    val.epoch = 0
    classes = weights['semantic']['classes']
    counts = np.zeros(len(classes), dtype=np.int64)
    for row in train.rows:
        mask = np.array(Image.open(args.data / row['mask']))
        counts += np.bincount(mask.ravel(), minlength=len(classes))
    class_weights = torch.tensor((counts.sum() / np.maximum(1, counts)) ** .5,
                                 device=device, dtype=torch.float32)
    class_weights /= class_weights.mean()
    cache = cache_validation(val, backbone, device, args.batch_size)
    optimizer = torch.optim.AdamW([p for head in heads.values() for p in head.parameters()],
                                  lr=args.learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,
                                                            eta_min=args.learning_rate * .15)
    edge_weight = torch.tensor(12., device=device)
    provenance = {
        'manifest_sha256': metadata['manifest_sha256'],
        'augmentation_sha256': hashlib.sha256(Path(__file__).with_name('camera_augmentation.py').read_bytes()).hexdigest(),
        'source_weights_sha256': {filename: hashlib.sha256((args.source / filename).read_bytes()).hexdigest()
                                  for filename in ('adapter.pt', 'landmarks.pt')},
        'train_images': len(train), 'validation_images': len(val),
    }
    history = []
    best = {}
    for epoch in range(args.epochs + 1):
        if epoch:
            for head in heads.values():
                head.train()
            train.epoch = epoch
            for raw, mask, points in DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2):
                rgb, features = encode(backbone, raw, device)
                mask, points = mask.to(device).long(), points.to(device)
                optimizer.zero_grad(set_to_none=True)
                segment = semantic_loss(heads['semantic'](features, rgb), mask, class_weights)
                landmarks = heads['landmarks'](features, rgb)
                corners = heatmap_loss(landmarks[:, :1], points[:, :1])
                studs = heatmap_loss(landmarks[:, 1:2], points[:, 1:2])
                edge = F.binary_cross_entropy_with_logits(landmarks[:, 2], points[:, 2], pos_weight=edge_weight)
                (segment + 2 * corners + studs + edge).backward()
                optimizer.step()
            scheduler.step()
        metrics = validation(heads, cache, val.rows, weights['landmarks']['thresholds'], classes, device)
        row = dict(epoch=epoch, **metrics)
        history.append(row)
        for name, filename, key in [('semantic', 'adapter.pt', 'adapter'), ('landmarks', 'landmarks.pt', 'head')]:
            clean, augmented = (metrics[c][name] for c in ('clean', 'augmented'))
            score = (clean + augmented) / 2
            # Retain the initial weights if a candidate trades more than one
            # point of clean validation quality for robustness.
            if clean < history[0]['clean'][name] - .01 or score <= best.get(name, {}).get('score', -1):
                continue
            best[name] = {'epoch': epoch, 'score': score}
            state = {**weights[name], key: heads[name].state_dict()}
            state.update(finetune_epoch=epoch, finetune_seed=args.seed)
            torch.save(state, args.output / filename)
        report = {'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
                  'learning_rate': args.learning_rate, 'source': str(args.source), 'best': best,
                  'validation': history, 'provenance': provenance}
        (args.output / 'finetune.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    report['threshold_calibration'] = calibrate_landmarks(
        args.output, args.source, heads['landmarks'], cache, val.rows, history[0]['clean']['landmark_f1'], device)
    (args.output / 'finetune.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
