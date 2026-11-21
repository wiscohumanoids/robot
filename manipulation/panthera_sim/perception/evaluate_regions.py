"""Compare connected and RGB-split block boxes against visible instance masks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from adapter import DenseAdapter, normalize, patch_features
from backbone import load_backbone
from evaluate_instances import box_iou, matched_pairs
from instances import extract_instances


def summarize(counts, overlaps):
    matched, predicted, truth = counts
    precision, recall = matched / max(1, predicted), matched / max(1, truth)
    return {'matched_blocks': matched, 'predicted_blocks': predicted, 'true_blocks': truth,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(1e-12, precision + recall),
            'matched_mean_box_iou': float(np.mean(overlaps)) if overlaps else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--predictions', type=Path, help='Read/write semantic PNG cache, named like manifest masks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest_bytes = (args.data / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if checkpoint['classes'] not in (['background', 'body', 'stud'], ['background', 'block_body', 'stud']):
        raise ValueError('Region evaluation requires background, body and stud classes')
    entries = manifest['splits'][args.split]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = backbone = None
    totals = {method: [0, 0, 0] for method in ('connected', 'rgb_split')}
    overlaps = {method: [] for method in totals}
    cameras = {method: {} for method in totals}
    frames = []
    cache_contract = {'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'split': args.split}
    cache_valid = False
    if args.predictions:
        args.predictions.mkdir(parents=True, exist_ok=True)
        cache_metadata = args.predictions / 'cache.json'
        cache_valid = cache_metadata.exists() and json.loads(cache_metadata.read_text()) == cache_contract
    for entry in entries:
        image = Image.open(args.data / entry['image']).convert('RGB')
        rgb_array = np.array(image)
        cached = args.predictions / Path(entry['mask']).name if args.predictions else None
        if cache_valid and cached and cached.exists():
            mask = np.array(Image.open(cached))
        else:
            if model is None:
                model = DenseAdapter(checkpoint['feature_dim'], len(checkpoint['classes'])).to(device).eval()
                model.load_state_dict(checkpoint['adapter'])
                backbone = load_backbone().to(device).eval()
            resized = image.resize(tuple(checkpoint['input_size'][::-1]), Image.Resampling.BILINEAR)
            rgb = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1)[None].float().to(device) / 255
            with torch.inference_mode():
                logits = model(patch_features(backbone, normalize(rgb)), rgb)
                mask = torch.nn.functional.interpolate(logits, size=rgb_array.shape[:2], mode='bilinear', align_corners=False).argmax(1)[0].cpu().numpy().astype('uint8')
            if cached:
                Image.fromarray(mask).save(cached)
        truth = np.array(Image.open(args.data / entry['instance']))
        true_boxes = []
        for instance_id in np.unique(truth[truth > 0]):
            y, x = np.nonzero(truth == instance_id)
            true_boxes.append([int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)])
        frame = {'image': entry['image'], 'camera_name': entry['camera_name']}
        for method in totals:
            objects = extract_instances(mask, rgb_array if method == 'rgb_split' else None)
            matrix = np.array([[box_iou(p['bbox_xyxy'], t) for t in true_boxes] for p in objects]).reshape(len(objects), len(true_boxes))
            pairs = matched_pairs(matrix, .5)
            counts = [len(pairs), len(objects), len(true_boxes)]
            for i, value in enumerate(counts):
                totals[method][i] += value
            frame_ious = [float(matrix[p, t]) for p, t in pairs]
            overlaps[method].extend(frame_ious)
            camera_counts, camera_ious = cameras[method].setdefault(entry['camera_name'], ([0, 0, 0], []))
            for i, value in enumerate(counts):
                camera_counts[i] += value
            camera_ious.extend(frame_ious)
            frame[method] = summarize(counts, frame_ious)
        frames.append(frame)
    if args.predictions:
        cache_metadata.write_text(json.dumps(cache_contract, indent=2) + '\n')
    report = {'data': str(args.data), 'split': args.split, 'frame_count': len(entries),
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': cache_contract['checkpoint_sha256'],
              'checkpoint_epoch': checkpoint['epoch'],
              'training_manifest_sha256': checkpoint['manifest_sha256'],
              'evaluation_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'scene_seed_range': [min(e['scene_seed'] for e in entries), max(e['scene_seed'] for e in entries)],
              'method': 'Connected baseline: all eight-connected semantic foreground components, no area filtering. RGB split: per-component saturated hue seeds, minimum seed area 80 pixels at 512 x 512 scaled by image area; saturation > 0.2, hue boundaries 8/30/80/170/255/330 degrees, 3 x 3 median at 512 x 512, nearest spatial seed fill. GT boxes enclose all visible instance pixels. One-to-one assignment maximizes count at IoU >= 0.5, then summed IoU. RGB parameters selected on original validation split before inspecting fresh holdout metrics.',
              'methods': {method: {**summarize(totals[method], overlaps[method]),
                                  'per_camera': {camera: summarize(*values) for camera, values in cameras[method].items()}}
                          for method in totals}, 'frames': frames}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
