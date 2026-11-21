"""Train landmark heatmaps with frozen DINOv3 features and tune on validation only."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from adapter import normalize, patch_features
from backbone import load_backbone, MODEL_NAME, MODEL_REVISION
from landmarks import LandmarkHead, decode_landmarks


def point_heatmap(points, height, width, sigma=2):
    result = np.zeros((height, width), np.float32)
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < width and 0 <= y < height):
            continue
        left, right = max(0, x - 6), min(width, x + 7)
        top, bottom = max(0, y - 6), min(height, y + 7)
        yy, xx = np.mgrid[top:bottom, left:right]
        result[top:bottom, left:right] = np.maximum(
            result[top:bottom, left:right],
            np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)),
        )
    return result


def heatmap_loss(logits, target):
    probability = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = target.eq(1)
    negative = target.lt(1)
    return (
        -(1 - probability).pow(2) * probability.log() * positive
        - probability.pow(2) * (1 - target).pow(4) * (1 - probability).log() * negative
    ).sum() / positive.sum().clamp_min(1)


def point_counts(prediction, truth, tolerance=5):
    pred = np.asarray([p["xy"] for p in prediction]).reshape(-1, 2)
    truth = np.asarray(truth).reshape(-1, 2)
    errors = []
    if len(pred) and len(truth):
        distances = np.linalg.norm(pred[:, None] - truth[None], axis=2)
        row, column = linear_sum_assignment(
            np.where(distances <= tolerance, distances, 1e6)
        )
        errors = [
            float(distances[r, c])
            for r, c in zip(row, column)
            if distances[r, c] <= tolerance
        ]
    return [len(errors), len(pred), len(truth)], errors


def summarize(counts, errors=None):
    matched, predictions, targets = counts
    precision, recall = matched / max(1, predictions), matched / max(1, targets)
    result = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "matched": matched,
        "predictions": predictions,
        "targets": targets,
    }
    if errors is not None:
        result.update(
            mean_localization_error_px=float(np.mean(errors)) if errors else None,
            median_localization_error_px=float(np.median(errors)) if errors else None,
        )
    return result


def evaluate(logits, records, edge_truths, thresholds):
    counts = {name: np.zeros(3, np.int64) for name in ["corners", "stud_centers"]}
    errors = {name: [] for name in counts}
    edge_counts = np.zeros(4, np.int64)
    for output, record, truth in zip(logits, records, edge_truths):
        decoded = decode_landmarks(output, thresholds)
        for name in counts:
            count, error = point_counts(decoded[name], record[name])
            counts[name] += count
            errors[name].extend(error)
        pred = decoded["edges"]
        truth = truth.astype(bool)
        edge_counts += [
            (
                np.count_nonzero(pred & (distance_transform_edt(~truth) <= 2))
                if truth.any()
                else 0
            ),
            pred.sum(),
            (
                np.count_nonzero(truth & (distance_transform_edt(~pred) <= 2))
                if pred.any()
                else 0
            ),
            truth.sum(),
        ]
    result = {name: summarize(counts[name].tolist(), errors[name]) for name in counts}
    precision = edge_counts[0] / max(1, edge_counts[1])
    recall = edge_counts[2] / max(1, edge_counts[3])
    result["edges"] = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "tolerance_px": 2,
    }
    return result


def predict(model, data, device, batch_size):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for rgb, _, features in DataLoader(data, batch_size=batch_size):
            outputs.append(
                model(features.to(device).float(), rgb.to(device)).half().cpu()
            )
    return torch.cat(outputs)


def gallery(logits, records, data, path, thresholds, count=8):
    h, w = data.tensors[0].shape[-2:]
    canvas = Image.new("RGB", (w * 3, (h + 22) * min(count, len(records))), "white")
    draw = ImageDraw.Draw(canvas)
    for i in range(min(count, len(records))):
        rgb = (data.tensors[0][i].permute(1, 2, 0).numpy() * 255).astype("uint8")
        decoded = decode_landmarks(logits[i], thresholds)
        for col, title in enumerate(
            ["RGB", "Visible ground truth", "Learned predictions"]
        ):
            image = Image.fromarray(rgb.copy())
            d = ImageDraw.Draw(image)
            if col:
                edge = (
                    decoded["edges"]
                    if col == 2
                    else data.tensors[1][i, 2].numpy().astype(bool)
                )
                pixels = np.array(image)
                pixels[edge] = (255, 220, 0)
                image = Image.fromarray(pixels)
                d = ImageDraw.Draw(image)
                for key, color in [
                    ("corners", (0, 255, 80)),
                    ("stud_centers", (255, 50, 240)),
                ]:
                    points = (
                        records[i][key] if col == 1 else [p["xy"] for p in decoded[key]]
                    )
                    for x, y in points:
                        d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            y = i * (h + 22)
            draw.text((col * w + 4, y + 4), title, fill="black")
            canvas.paste(image, (col * w, y + 22))
    canvas.save(path)


def load_datasets(data_root, records, device, batch_size):
    """Rasterize point targets and cache frozen features once per scene."""
    datasets = {}
    edge_truths = {}
    backbone = load_backbone().to(device)
    for split, rows in records.items():
        images = []
        targets = []
        edges = []
        for row in rows:
            rgb = np.array(Image.open(data_root / row["image"]).convert("RGB"))
            h, w = rgb.shape[:2]
            edge = np.array(Image.open(data_root / row["edges"]).convert("L")) > 0
            images.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255)
            targets.append(
                torch.from_numpy(
                    np.stack(
                        [
                            point_heatmap(row["corners"], h, w),
                            point_heatmap(row["stud_centers"], h, w),
                            edge.astype(np.float32),
                        ]
                    )
                )
            )
            edges.append(edge)
        rgb = torch.stack(images)
        features = []
        with torch.inference_mode():
            for batch in rgb.split(batch_size):
                features.append(
                    patch_features(backbone, normalize(batch.to(device))).half().cpu()
                )
        datasets[split] = TensorDataset(rgb, torch.stack(targets), torch.cat(features))
        edge_truths[split] = edges
        print(f"{split}: {len(rows)} scenes cached", flush=True)
    del backbone
    torch.cuda.empty_cache()
    return datasets, edge_truths


def calibrate_thresholds(val_logits, records, edge_truths):
    """Choose each detection threshold by validation F1."""
    thresholds = {"corner": 0.2, "stud_center": 0.2, "body_edge": 0.5}
    sweeps = {}
    # Each channel threshold uses only its validation metric; test is evaluated once.
    for name, key in [
        ("corner", "corners"),
        ("stud_center", "stud_centers"),
        ("body_edge", "edges"),
    ]:
        results = []
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            candidate = dict(thresholds)
            candidate[name] = threshold
            metric = evaluate(val_logits, records, edge_truths, candidate)[key]
            results.append({"threshold": threshold, **metric})
        sweeps[name] = results
        thresholds[name] = max(results, key=lambda r: r["f1"])["threshold"]
    return thresholds, sweeps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    source = (args.data / "landmarks.json").read_bytes()
    metadata = json.loads(source)
    frames = metadata["frames"]
    records = {
        split: [r for r in frames if r["split"] == split]
        for split in ["train", "val", "test"]
    }
    start = time.monotonic()
    datasets, edge_truths = load_datasets(args.data, records, device, args.batch_size)
    model = LandmarkHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0.00005
    )
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    history = []
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for rgb, target, features in loader:
            rgb, target, features = (
                rgb.to(device),
                target.to(device),
                features.to(device).float(),
            )
            if random.random() < 0.5:
                rgb, target, features = rgb.flip(-1), target.flip(-1), features.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            output = model(features, rgb)
            points = 2 * heatmap_loss(output[:, :1], target[:, :1]) + heatmap_loss(
                output[:, 1:2], target[:, 1:2]
            )
            edge = F.binary_cross_entropy_with_logits(
                output[:, 2], target[:, 2], pos_weight=torch.tensor(12.0, device=device)
            )
            loss = points + edge
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        scheduler.step()
        val_logits = predict(model, datasets["val"], device, args.batch_size)
        val_target = datasets["val"].tensors[1]
        val_loss = float(
            2 * heatmap_loss(val_logits[:, :1].float(), val_target[:, :1])
            + heatmap_loss(val_logits[:, 1:2].float(), val_target[:, 1:2])
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_point_loss": val_loss,
            }
        )
        print(
            f"epoch {epoch+1}/{args.epochs} train={np.mean(losses):.4f} val_point={val_loss:.4f}",
            flush=True,
        )
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "head": model.state_dict(),
                    "feature_dim": 384,
                    "epoch": epoch + 1,
                    "input_size": list(datasets["train"].tensors[0].shape[-2:]),
                    "backbone": MODEL_NAME,
                    "backbone_revision": MODEL_REVISION,
                    "landmarks_sha256": hashlib.sha256(source).hexdigest(),
                },
                args.output / "landmarks_training.pt",
            )
        (args.output / "landmark_history.json").write_text(
            json.dumps(history, indent=2)
        )
    checkpoint = torch.load(
        args.output / "landmarks_training.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["head"])
    val_logits = predict(model, datasets["val"], device, args.batch_size)
    thresholds, sweeps = calibrate_thresholds(
        val_logits, records["val"], edge_truths["val"]
    )
    checkpoint["thresholds"] = thresholds
    torch.save(checkpoint, args.output / "landmarks.pt")
    test_logits = predict(model, datasets["test"], device, args.batch_size)
    report = {
        "validation": evaluate(
            val_logits, records["val"], edge_truths["val"], thresholds
        ),
        "test": evaluate(test_logits, records["test"], edge_truths["test"], thresholds),
        "thresholds": thresholds,
        "validation_threshold_sweeps": sweeps,
        "configuration": {
            "epochs": args.epochs,
            "best_epoch": checkpoint["epoch"],
            "seed": 42,
            "backbone_frozen": True,
            "point_tolerance_px_at_512": 5,
            "edge_tolerance_px_at_512": 2,
            "matching": "one-to-one minimum distance assignment with distances above tolerance excluded",
            "split_sizes": {s: len(r) for s, r in records.items()},
            "seconds": time.monotonic() - start,
            "landmarks_sha256": checkpoint["landmarks_sha256"],
        },
    }
    report["test_by_camera"] = {}
    for camera in sorted({r.get("camera_name", "unknown") for r in records["test"]}):
        indices = [
            i
            for i, r in enumerate(records["test"])
            if r.get("camera_name", "unknown") == camera
        ]
        report["test_by_camera"][camera] = evaluate(
            test_logits[indices],
            [records["test"][i] for i in indices],
            [edge_truths["test"][i] for i in indices],
            thresholds,
        )
    (args.output / "landmark_metrics.json").write_text(json.dumps(report, indent=2))
    gallery(
        test_logits,
        records["test"],
        datasets["test"],
        args.output / "landmark_test_predictions.jpg",
        thresholds,
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "validation_threshold_sweeps"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
