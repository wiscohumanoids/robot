"""Evaluate the pickup detector on paired clean and camera-augmented RGB frames."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .camera_augmentation import CONDITIONS, augment_rgb
from .evaluate_instances import box_iou, matched_pairs
from .pickup import DinoPickupPerception, detect_blocks


def image_targets(entry, instance, landmarks):
    colors = {block['instance_id']: block['color'] for block in entry['blocks']}
    targets = []
    for idx in np.unique(instance[instance > 0]):
        y, x = np.nonzero(instance == idx)
        targets.append((colors[idx], [x.min(), y.min(), x.max() + 1, y.max() + 1],
                        [s['center_xy'] for s in landmarks['studs'] if s['instance_id'] == idx]))
    return targets


def point_errors(pred, truth):
    pred = np.array(pred).reshape(-1, 2)
    truth = np.array(truth).reshape(-1, 2)
    distance = np.linalg.norm(pred[:, None] - truth[None, :], axis=2)
    return [distance[i, j] for i, j in matched_pairs(distance, 5., maximize=False)]


def score_image(detector, rgb, mask, targets, landmarks):
    probs, corners, studs = detector._predict_frame(rgb, include_landmarks=True)
    blocks = detect_blocks(rgb, probs, corners_xy=corners, stud_centers_xy=studs)
    pred = probs.argmax(0)
    confusion = np.bincount((mask * 3 + pred).ravel(), minlength=9)
    overlap = np.array([[box_iou(block.bbox_xyxy, box) for _, box, _ in targets]
                        for block in blocks]).reshape(len(blocks), len(targets))
    count = len(matched_pairs(overlap, .5))
    for i, block in enumerate(blocks):
        for j, (color, _, _) in enumerate(targets):
            if color != block.color:
                overlap[i, j] = 0.
    matched = matched_pairs(overlap, .5)
    errors = []
    for i, j in matched:
        errors.extend(point_errors(blocks[i].studs_xy, targets[j][2]))
    metrics = {'confusion': confusion.tolist(), 'blocks': [count, len(blocks), len(targets)],
               'color_blocks': [len(matched), len(blocks), len(targets)],
               'studs_on_color_blocks': [len(errors), sum(len(b.studs_xy) for b in blocks),
                                         sum(len(points) for _, _, points in targets)],
               'sum_stud_error_px': float(sum(errors))}
    for name, points, truth in [('raw_studs', studs, landmarks['stud_centers']),
                                ('raw_corners', corners, landmarks['corners'])]:
        metrics[name] = [len(point_errors(points, truth)), len(points), len(truth)]
    return metrics


def summarize(counts):
    matrix = counts['confusion'].reshape(*counts['confusion'].shape[:-1], 3, 3)
    diagonal = matrix.diagonal(axis1=-2, axis2=-1)
    iou = diagonal / np.maximum(matrix.sum(-1) + matrix.sum(-2) - diagonal, 1)
    summary = {'foreground_miou': iou[..., 1:].mean(-1),
               'body_iou': iou[..., 1], 'stud_iou': iou[..., 2]}
    for key in ('blocks', 'color_blocks', 'studs_on_color_blocks', 'raw_studs', 'raw_corners'):
        matched, predicted, target = np.moveaxis(counts[key], -1, 0)
        summary[key + '_precision'] = matched / np.maximum(predicted, 1)
        summary[key + '_recall'] = matched / np.maximum(target, 1)
        summary[key + '_f1'] = 2 * matched / np.maximum(predicted + target, 1)
    return summary


def grouped_counts(frames, scenes):
    grouped = {}
    for key in frames[0]['counts']:
        values = np.asarray([f['counts'][key] for f in frames])
        groups = np.zeros((len(scenes), *values.shape[1:]), dtype=values.dtype)
        np.add.at(groups, np.searchsorted(scenes, [f['scene_seed'] for f in frames]), values)
        grouped[key] = groups
    return grouped


def compare(frames, reference, weights):
    scenes = np.unique([f['scene_seed'] for f in frames])
    if [(f['image'], f['scene_seed']) for f in frames] != [(f['image'], f['scene_seed']) for f in reference]:
        raise ValueError('Pairing requires the same scenes and image order')
    def collect(rows):
        counts = grouped_counts(rows, scenes)
        metrics = summarize({k: a.sum(0) for k, a in counts.items()})
        resampled = summarize({k: weights @ a for k, a in counts.items()})
        return metrics, resampled
    observed, bootstrap = collect(frames)
    prior, bootstrap_prior = collect(reference)
    return {k: {'delta': float(observed[k] - prior[k]), 'ci95':
                np.quantile(bootstrap[k] - bootstrap_prior[k], [.025, .975]).tolist()}
            for k in observed}


def totals(frames):
    counts = {k: np.sum([f['counts'][k] for f in frames], axis=0) for k in frames[0]['counts']}
    report = {name: float(value) for name, value in summarize(counts).items()}
    matches = counts['studs_on_color_blocks'][0]
    report['matched_stud_error_px'] = float(counts['sum_stud_error_px'] / matches) if matches else None
    return report


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/megabloks'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--landmarks', type=Path, help='Defaults to landmarks.pt beside the semantic checkpoint')
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--reference', type=Path, help='Earlier report for paired checkpoint comparisons')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = args.data / 'manifest.json'
    labels = args.data / 'landmarks.json'
    sidecar = json.loads(labels.read_text())
    if sidecar['manifest_sha256'] != sha(manifest):
        raise ValueError('Landmark truth belongs to a different image manifest')
    truth = {entry['image']: entry for entry in sidecar['frames']}
    entries = json.loads(manifest.read_text())['splits'][args.split]
    landmark_path = args.landmarks or args.checkpoint.with_name('landmarks.pt')
    detector = DinoPickupPerception(args.checkpoint, landmark_checkpoint=landmark_path)
    scenes = np.unique([entry['scene_seed'] for entry in entries])
    protocol = dict(split=args.split, seed=args.seed, manifest_sha256=sha(manifest),
                    landmark_labels_sha256=sha(labels), conditions=list(CONDITIONS),
                    augmentation_sha256=sha(Path(__file__).with_name('camera_augmentation.py')))
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and reference['protocol'] != protocol:
        raise ValueError('Comparison report has different data or augmentation settings')
    report = dict(protocol=protocol, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
                  landmarks=str(landmark_path), landmarks_sha256=sha(landmark_path), frames=len(entries),
                  scenes=len(scenes), conditions={}, reference=str(args.reference) if reference else None,
                  scoring={'box_iou_min': .5, 'stud_distance_max_px': 5.,
                           'stud_assignment': 'One-to-one within one-to-one, color-correct matched boxes',
                           'foreground_miou': 'Mean of pooled body and stud class IoUs',
                           'ci95': 'Percentile interval over 2000 paired bootstrap samples of scenes'})
    # Both members of a pair receive identical draws of complete scenes.
    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(len(scenes), np.full(len(scenes), 1 / len(scenes)), size=2000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for condition in dict.fromkeys(('clean', 'mixed', *CONDITIONS)):
        frames = []
        for entry in entries:
            rgb = np.asarray(Image.open(args.data / entry['image']).convert('RGB'))
            rng = np.random.default_rng([args.seed, entry['scene_seed'], CONDITIONS.index(condition)])
            image = augment_rgb(rgb, condition, rng)
            mask = np.asarray(Image.open(args.data / entry['mask']))
            instance = np.asarray(Image.open(args.data / entry['instance']))
            landmarks = truth[entry['image']]
            targets = image_targets(entry, instance, landmarks)
            counts = score_image(detector, image, mask, targets, landmarks)
            frames.append(dict(image=entry['image'], camera=entry['camera_name'],
                               scene_seed=entry['scene_seed'], counts=counts))
        summary = totals(frames)
        section = {'metrics': summary, 'images': frames, 'per_camera': {}}
        for camera in sorted({frame['camera'] for frame in frames}):
            section['per_camera'][camera] = totals([frame for frame in frames if frame['camera'] == camera])
        report['conditions'][condition] = section
        if condition != 'clean':
            section['vs_clean'] = compare(frames, report['conditions']['clean']['images'], weights)
        if reference:
            section['vs_reference'] = compare(frames, reference['conditions'][condition]['images'], weights)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(condition, {k: round(summary[k], 4) for k in ('foreground_miou', 'color_blocks_f1',
                                                             'studs_on_color_blocks_f1', 'raw_studs_f1')},
              flush=True)


if __name__ == '__main__':
    main()
