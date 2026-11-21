"""Segment block bodies and studs in RGB images with a trained DINOv3 adapter."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from adapter import DenseAdapter, normalize, patch_features
from backbone import MODEL_NAME, MODEL_REVISION, load_backbone
from train import palette
from instances import annotate_instances, instance_regions, measure_instances


def associate_block(xy, objects, regions=None):
    """Use nearby region pixels, then unambiguous box containment, for point IDs."""
    x, y = xy
    if regions is not None:
        ix, iy = round(x), round(y)
        nearby = regions[max(0, iy-3):iy+4, max(0, ix-3):ix+4]
        ids = np.unique(nearby[nearby > 0])
        if len(ids) == 1:
            return int(ids[0])
    candidates = [obj['id'] for obj in objects
                  if obj['bbox_xyxy'][0] - 3 <= x < obj['bbox_xyxy'][2] + 3
                  and obj['bbox_xyxy'][1] - 3 <= y < obj['bbox_xyxy'][3] + 3]
    return candidates[0] if len(candidates) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instance-method', choices=['rgb', 'connected'], default='rgb',
                        help='Split touching plastic colors with RGB, or use connected semantic foreground')
    parser.add_argument('--landmark-checkpoint', type=Path, help='Optional corner, stud-center and edge head; defaults to sibling landmarks.pt')
    parser.add_argument('--size', type=int, help='Override the checkpoint input resolution')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint['backbone'] != MODEL_NAME or checkpoint['backbone_revision'] != MODEL_REVISION:
        raise ValueError('The checkpoint backbone name or revision does not match the loader')
    input_size = [args.size, args.size] if args.size is not None else checkpoint['input_size']
    if any(size <= 0 or size % 16 for size in input_size):
        parser.error('Input dimensions must be positive and divisible by 16')
    model = DenseAdapter(checkpoint['feature_dim'], len(checkpoint['classes'])).to(device).eval()
    model.load_state_dict(checkpoint['adapter'])
    backbone = load_backbone().to(device).eval()
    landmark_path = args.landmark_checkpoint or args.checkpoint.with_name('landmarks.pt')
    landmark_model = None
    if landmark_path.exists() and checkpoint['classes'] in (['background', 'body', 'stud'], ['background', 'block_body', 'stud']):
        from landmarks import LandmarkHead, decode_landmarks
        landmark_checkpoint = torch.load(landmark_path, map_location=device, weights_only=True)
        if landmark_checkpoint['backbone'] != MODEL_NAME:
            raise ValueError('The landmark checkpoint uses a different backbone')
        if landmark_checkpoint.get('backbone_revision', MODEL_REVISION) != MODEL_REVISION:
            raise ValueError('The landmark checkpoint uses a different backbone revision')
        landmark_model = LandmarkHead(landmark_checkpoint['feature_dim']).to(device).eval()
        landmark_model.load_state_dict(landmark_checkpoint['head'])
    elif args.landmark_checkpoint is not None:
        raise ValueError('A landmark checkpoint requires a matching Mega Bloks semantic checkpoint and an existing file')
    image = Image.open(args.image).convert('RGB')
    resized = image.resize(tuple(input_size[::-1]), Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1)[None].float().to(device) / 255
    with torch.inference_mode():
        features = patch_features(backbone, normalize(rgb))
        logits = model(features, rgb)
        landmark_logits = None
        if landmark_model is not None:
            landmark_network_logits = landmark_model(features, rgb)
            landmark_logits = torch.nn.functional.interpolate(landmark_network_logits, size=(image.height, image.width), mode='bilinear', align_corners=False)[0]
        logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode='bilinear', align_corners=False)
        probabilities = logits.softmax(1)[0].cpu().numpy()
        mask = probabilities.argmax(0).astype('uint8')
    Image.fromarray(mask).save(args.output / 'semantic.png')
    colors = palette(len(checkpoint['classes']))
    overlay = np.array(image).copy()
    foreground = mask > 0
    overlay[foreground] = (.5 * overlay[foreground] + .5 * colors[mask[foreground]]).astype('uint8')
    Image.fromarray(overlay).save(args.output / 'overlay.png')
    objects = None
    regions = None
    if checkpoint['classes'] in (['background', 'body', 'stud'], ['background', 'block_body', 'stud']):
        instance_rgb = np.array(image) if args.instance_method == 'rgb' else None
        regions = instance_regions(mask, rgb=instance_rgb)
        objects = measure_instances(mask, regions)
        Image.fromarray(regions.astype('uint16')).save(args.output / 'instance_ids.png')
        annotate_instances(Image.fromarray(overlay), objects).save(args.output / 'instances.png')
    np.savez_compressed(args.output / 'probabilities.npz', probabilities=probabilities)
    summary = {'image': str(args.image), 'image_size_wh': [image.width, image.height],
               'coordinate_space': 'original RGB pixels; origin top-left; x right, y down',
               'classes': checkpoint['classes'],
               'pixels': {name: int((mask == i).sum()) for i, name in enumerate(checkpoint['classes'])}}
    if objects is not None:
        summary['blocks'] = objects
        summary['instance_method'] = args.instance_method
    if landmark_logits is not None:
        decoded = decode_landmarks(landmark_network_logits[0], landmark_checkpoint['thresholds'])
        decoded.pop('edges')
        landmark_probabilities = landmark_logits.sigmoid().cpu().numpy()
        edges = landmark_probabilities[2] >= landmark_checkpoint['thresholds']['body_edge']
        for points in decoded.values():
            for point in points:
                x, y = point['xy']
                point['xy'] = [(x + .5) * image.width / input_size[1] - .5,
                               (y + .5) * image.height / input_size[0] - .5]
                point['block_id'] = associate_block(point['xy'], objects or [], regions)
        summary['landmarks'] = decoded
        summary['landmark_checkpoint'] = str(landmark_path)
        summary['landmark_thresholds'] = landmark_checkpoint['thresholds']
        np.savez_compressed(args.output / 'landmark_probabilities.npz', probabilities=landmark_probabilities,
                            channels=np.array(['corner', 'stud_center', 'body_edge']))
        for channel, name in enumerate(['corner_heatmap', 'stud_center_heatmap', 'edge_probability']):
            Image.fromarray((landmark_probabilities[channel] * 255).astype('uint8')).save(args.output / f'{name}.png')
        Image.fromarray(edges.astype('uint8') * 255).save(args.output / 'edges.png')
        combined = overlay.copy()
        combined[edges] = [0, 255, 255]
        combined = Image.fromarray(combined)
        if objects:
            combined = annotate_instances(combined, [{**obj, 'studs': []} for obj in objects])
        draw = ImageDraw.Draw(combined)
        for point in decoded['corners']:
            x, y = point['xy']
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 140, 0), outline=(20, 20, 20), width=1)
        for point in decoded['stud_centers']:
            x, y = point['xy']
            draw.line((x-4, y, x+4, y), fill=(50, 255, 50), width=2)
            draw.line((x, y-4, x, y+4), fill=(50, 255, 50), width=2)
        combined.save(args.output / 'combined.png')
    (args.output / 'prediction.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
