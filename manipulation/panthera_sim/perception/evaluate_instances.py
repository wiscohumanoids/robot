"""Evaluate visible block boxes and stud centers on an untouched test split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F


def matched_pairs(values, threshold, maximize=True):
    """Maximize valid match count, then favor larger IoU or smaller distance."""
    if not values.size:
        return []
    valid = values >= threshold if maximize else values <= threshold
    quality = values if maximize else 1 - values / threshold
    reward = valid * (min(values.shape) + 1 + quality)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.


def main():
    from adapter import DenseAdapter, normalize, patch_features
    from backbone import load_backbone
    from instances import extract_instances

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    manifest_bytes = (args.data / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if checkpoint['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('Checkpoint and evaluation manifest differ')
    stud_path = next((args.data / name for name in ['landmarks.json', 'test_landmarks.json', 'test_stud_centers.json']
                      if (args.data / name).exists()), None)
    if stud_path is None:
        raise FileNotFoundError('Run tools/replay_mega_bloks_landmarks.py to produce physical stud labels')
    stud_truth = json.loads(stud_path.read_text())
    if stud_truth['manifest_sha256'] != checkpoint['manifest_sha256']:
        raise ValueError('Physical-stud truth and checkpoint manifests differ')
    stud_frames = {frame['image']: frame['studs'] for frame in stud_truth['frames']}
    model = DenseAdapter(checkpoint['feature_dim']).to(device).eval()
    model.load_state_dict(checkpoint['adapter'])
    backbone = load_backbone().to(device).eval()
    counts = dict(predicted_blocks=0, true_blocks=0, matched_blocks=0,
                  predicted_studs=0, true_studs=0, matched_studs=0)
    ious, errors, images = [], [], []
    cameras = {}
    for entry in manifest['splits']['test']:
        before = counts.copy()
        image = Image.open(args.data / entry['image']).convert('RGB')
        truth = np.array(Image.open(args.data / entry['mask']))
        instance = np.array(Image.open(args.data / entry['instance']))
        rgb = image.resize(tuple(reversed(checkpoint['input_size'])), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)[None].float().to(device) / 255
        with torch.inference_mode():
            logits = model(patch_features(backbone, normalize(rgb)), rgb)
            prediction = F.interpolate(logits, size=truth.shape, mode='bilinear', align_corners=False).argmax(1)[0].cpu().numpy()
        predicted = extract_instances(prediction)
        targets = []
        for instance_id in np.unique(instance[instance > 0]):
            region = instance == instance_id
            ys, xs = np.nonzero(region)
            targets.append({'bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                            'studs': [{'centroid_xy': stud['center_xy']} for stud in stud_frames[entry['image']]
                                      if stud['instance_id'] == int(instance_id)]})
        matrix = np.array([[box_iou(p['bbox_xyxy'], t['bbox_xyxy']) for t in targets] for p in predicted]).reshape(len(predicted), len(targets))
        pairs = matched_pairs(matrix, .5)
        counts['predicted_blocks'] += len(predicted)
        counts['true_blocks'] += len(targets)
        counts['matched_blocks'] += len(pairs)
        counts['predicted_studs'] += sum(len(p['studs']) for p in predicted)
        counts['true_studs'] += sum(len(t['studs']) for t in targets)
        for p, t in pairs:
            ious.append(float(matrix[p, t]))
            ps = np.array([s['centroid_xy'] for s in predicted[p]['studs']]).reshape(-1, 2)
            ts = np.array([s['centroid_xy'] for s in targets[t]['studs']]).reshape(-1, 2)
            distances = np.linalg.norm(ps[:, None] - ts[None, :], axis=2)
            stud_pairs = matched_pairs(distances, 5., maximize=False)
            counts['matched_studs'] += len(stud_pairs)
            errors.extend(float(distances[a, b]) for a, b in stud_pairs)
        camera = entry['camera_name']
        camera_counts = cameras.setdefault(camera, {k: 0 for k in counts})
        frame_counts = {k: counts[k] - before[k] for k in counts}
        for key, value in frame_counts.items():
            camera_counts[key] += value
        images.append({'image': entry['image'], 'camera_name': camera, **frame_counts})
    report = {'checkpoint': str(args.checkpoint), 'epoch': checkpoint['epoch'],
              'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'split': 'test',
              'method': 'All 8-connected predicted foreground regions; no size filtering. GT boxes enclose all visible pixels per instance ID. One-to-one assignment maximizes number of box IoU >= 0.5 matches, then summed IoU. GT stud centers average visible pixels per physical MuJoCo stud geom, from replay landmark labels; predicted centers are connected stud-label components. Within matched blocks, one-to-one centroid assignment maximizes matches within 5 pixels, then minimizes distance. Unmatched blocks contribute unmatched studs. Thresholds fixed before evaluation; no test tuning.',
              'counts': counts, 'images': images,
              'matched_box_mean_iou': float(np.mean(ious)) if ious else None,
              'matched_stud_mean_error_pixels': float(np.mean(errors)) if errors else None}
    for kind in ['blocks', 'studs']:
        report[kind] = {'precision': counts['matched_' + kind] / max(1, counts['predicted_' + kind]),
                        'recall': counts['matched_' + kind] / max(1, counts['true_' + kind])}
    report['per_camera'] = {}
    for camera, camera_counts in cameras.items():
        report['per_camera'][camera] = {'counts': camera_counts}
        for kind in ['blocks', 'studs']:
            report['per_camera'][camera][kind] = {
                'precision': camera_counts['matched_' + kind] / max(1, camera_counts['predicted_' + kind]),
                'recall': camera_counts['matched_' + kind] / max(1, camera_counts['true_' + kind])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'images'}, indent=2))


if __name__ == '__main__':
    main()
