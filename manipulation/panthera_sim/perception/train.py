"""Train and evaluate a dense adapter while keeping DINOv3 frozen."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from adapter import DenseAdapter, normalize, patch_features
from backbone import MODEL_NAME, MODEL_REVISION, load_backbone

COLORS = np.array([[35, 35, 35], [35, 160, 245], [255, 70, 140]], dtype=np.uint8)


def palette(num_classes):
    if num_classes == 3:
        return COLORS
    rng = np.random.default_rng(7)
    colors = rng.integers(45, 256, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = 35
    return colors


def read_split(root, entries):
    images, masks = [], []
    for entry in entries:
        images.append(torch.from_numpy(np.array(Image.open(root / entry['image']).convert('RGB'), copy=True)).permute(2, 0, 1).float() / 255)
        masks.append(torch.from_numpy(np.array(Image.open(root / entry['mask']), copy=True)).long())
    return torch.stack(images), torch.stack(masks)


def confusion(logits, labels):
    num_classes = logits.shape[1]
    pred = logits.argmax(1)
    return torch.bincount((labels * num_classes + pred).flatten(), minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu()


def scores(matrix, class_names=None):
    class_names = class_names or ['background', 'block_body', 'stud']
    m = matrix.double()
    iou = m.diag() / (m.sum(0) + m.sum(1) - m.diag()).clamp_min(1)
    precision = m.diag() / m.sum(0).clamp_min(1)
    recall = m.diag() / m.sum(1).clamp_min(1)
    foreground_intersection = m[1:, 1:].sum()
    foreground_union = m[1:, :].sum() + m[:, 1:].sum() - foreground_intersection
    return {'foreground_union_iou': (foreground_intersection / foreground_union.clamp_min(1)).item(),
            'mean_iou': iou.mean().item(), 'foreground_mean_iou': iou[1:].mean().item(),
            'pixel_accuracy': (m.diag().sum() / m.sum()).item(),
            'classes': {name: {'iou': iou[i].item(), 'precision': precision[i].item(), 'recall': recall[i].item()}
                        for i, name in enumerate(class_names)}, 'confusion_matrix': matrix.tolist()}


def evaluate(model, data, device, batch_size, class_names):
    matrix = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for rgb, target, features in DataLoader(data, batch_size=batch_size):
            logits = model(features.to(device).float(), rgb.to(device))
            matrix += confusion(logits, target.to(device))
    return scores(matrix, class_names)


def gallery(model, data, path, device, colors, count=12):
    model.eval()
    width, height = data.tensors[0].shape[-1], data.tensors[0].shape[-2]
    canvas = Image.new('RGB', (width * 3, (height + 24) * min(count, len(data))), 'white')
    draw = ImageDraw.Draw(canvas)
    with torch.inference_mode():
        for i in range(min(count, len(data))):
            rgb, truth, features = data[i]
            prediction = model(features[None].to(device).float(), rgb[None].to(device)).argmax(1)[0].cpu().numpy()
            y = i * (height + 24)
            for j, title in enumerate(['RGB', 'Ground truth', 'Prediction']):
                draw.text((j * width + 5, y + 5), title, fill='black')
            canvas.paste(Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).astype('uint8')), (0, y + 24))
            canvas.paste(Image.fromarray(colors[truth.numpy()]), (width, y + 24))
            canvas.paste(Image.fromarray(colors[prediction]), (width * 2, y + 24))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    manifest_bytes = (args.data / 'manifest.json').read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    splits = manifest.get('splits', manifest)
    classes = manifest['classes']
    class_names = [classes[str(i)] for i in range(len(classes))] if isinstance(classes, dict) else classes
    if not 2 <= len(class_names) <= 256:
        raise ValueError('The manifest must define 2–256 contiguous classes with background at ID 0')
    colors = palette(len(class_names))
    backbone = load_backbone().to(device).eval()
    datasets = {}
    start = time.monotonic()
    for split in ['train', 'val', 'test']:
        rgb, masks = read_split(args.data, splits[split])
        if masks.min() < 0 or masks.max() >= len(class_names):
            raise ValueError(f'{split}: mask IDs fall outside manifest classes')
        if any(n % 16 for n in rgb.shape[-2:]):
            raise ValueError('Image dimensions must be divisible by the 16-pixel patch stride')
        batches = []
        with torch.inference_mode():
            for batch in rgb.split(args.batch_size):
                batches.append(patch_features(backbone, normalize(batch.to(device))).half().cpu())
        features = torch.cat(batches)
        datasets[split] = TensorDataset(rgb, masks, features)
        print(f'{split}: {len(rgb)} images, features {tuple(features.shape)}', flush=True)
    del backbone
    torch.cuda.empty_cache()
    model = DenseAdapter(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    counts = torch.bincount(datasets['train'].tensors[1].flatten(), minlength=len(class_names)).float()
    weights = (counts.sum() / counts.clamp_min(1)).sqrt(); weights /= weights.mean()
    initial_metrics = evaluate(model, datasets['test'], device, args.batch_size, class_names)
    history = []
    best = -1
    loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0
        for rgb, target, features in loader:
            rgb, target, features = rgb.to(device), target.to(device), features.to(device).float()
            # Flip features and labels together, preserving patch alignment.
            if random.random() < .5:
                rgb, target, features = rgb.flip(-1), target.flip(-1), features.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, rgb)
            ce = F.cross_entropy(logits, target, weight=weights.to(device))
            probabilities = logits.softmax(1)
            onehot = F.one_hot(target, len(class_names)).permute(0, 3, 1, 2)
            dice = 1 - ((2 * (probabilities * onehot).sum((0, 2, 3)) + 1) /
                        (probabilities.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1)).mean()
            loss = ce + dice
            loss.backward(); optimizer.step()
            total += loss.item() * len(rgb)
        val = evaluate(model, datasets['val'], device, args.batch_size, class_names)
        row = {'epoch': epoch + 1, 'train_loss': total / len(datasets['train']), 'val': val}
        history.append(row)
        print(f"epoch {epoch + 1}/{args.epochs} loss={row['train_loss']:.4f} val_mIoU={val['mean_iou']:.4f} foreground_mIoU={val['foreground_mean_iou']:.4f}", flush=True)
        if val['foreground_mean_iou'] > best:
            best = val['foreground_mean_iou']
            torch.save({'adapter': model.state_dict(), 'feature_dim': 384, 'epoch': epoch + 1,
                        'classes': class_names, 'seed': args.seed,
                        'backbone': MODEL_NAME, 'backbone_revision': MODEL_REVISION,
                        'input_size': list(datasets['train'].tensors[0].shape[-2:]),
                        'manifest_sha256': manifest_hash}, args.output / 'adapter.pt')
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(args.output / 'adapter.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['adapter'])
    metrics = {split: evaluate(model, data, device, args.batch_size, class_names) for split, data in datasets.items()}
    metrics['untrained_adapter_test'] = initial_metrics
    metrics['configuration'] = {'seed': args.seed, 'epochs': args.epochs, 'best_epoch': checkpoint['epoch'],
                                'backbone_frozen': True, 'adapter_parameters': sum(p.numel() for p in model.parameters()),
                                'seconds': time.monotonic() - start, 'device': str(device), 'manifest_sha256': manifest_hash,
                                'backbone': MODEL_NAME, 'backbone_revision': MODEL_REVISION,
                                'split_sizes': {k: len(v) for k, v in datasets.items()}}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    gallery(model, datasets['test'], args.output / 'test_predictions.jpg', device, colors)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
