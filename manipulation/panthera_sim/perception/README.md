# Identify Mega Bloks, corners, edges and studs with DINOv3

Use the [two-arm Panthera wood-bench simulator](../sim/panthera/MEGA_BLOKS.md) to train two heads on frozen DINOv3 patch features and RGB. One segments the block bodies and studs. The other locates visible body corners, straight edges and individual stud centers. Both run on one RGB image with a single pass through the encoder.

## Install

Run these commands from `manipulation/panthera_sim`. The first install selects the PyTorch CUDA 12.8 wheels used for training on Linux with an NVIDIA GPU.

```bash
uv venv .venv-dino --python 3.12
uv pip install --python .venv-dino/bin/python torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-dino/bin/python -r perception/requirements.txt
```

The encoder loads the [timm conversion of DINOv3 ViT-S/16](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m) into the Hugging Face cache on first use. [The loader](backbone.py) pins the weight revision. The weights are distributed under [Meta's DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).

## Generate data and train

Skip generation when using the existing dataset. To generate another collection, choose a fresh output directory and pass it to the trainer.

```bash
.venv-dino/bin/python tools/generate_mega_bloks_segmentation.py --output data/megabloks
.venv-dino/bin/python perception/train.py --data data/megabloks \
    --output outputs/megabloks_dinov3 --batch-size 4
```

The generator writes RGB, semantic and instance PNGs and records their relative paths by split in `manifest.json`. MuJoCo supplies the visible geom ID for each labeled pixel. The semantic mask stores background as 0, body as 1 and stud as 2. The instance mask groups all visible surfaces of a block under one ID.

Training extracts patch features once per image and fits the decoder against the semantic masks. The run saves the adapter with the highest validation mean IoU across body and stud classes as `adapter.pt`. The split metrics go to `metrics.json`, and `test_predictions.jpg` compares held-out RGB, labels and predictions.

## Train the corner, edge and stud-center head

```bash
.venv-dino/bin/python tools/replay_mega_bloks_landmarks.py --dataset data/megabloks
.venv-dino/bin/python perception/train_landmarks.py --data data/megabloks \
    --output outputs/megabloks_dinov3 --batch-size 4
```

Skip the replay command if `landmarks.json` already exists. Replay projects the eight corners and twelve edges of each rectangular body through the camera used for that frame. Body IDs and rendered depth determine which corners and edge segments are visible. Each stud target is the centroid of the visible pixels of its own MuJoCo geom, including when neighboring studs touch in the image. This defines a 2D image landmark, which can move as the stud becomes occluded.

The trainer uses the validation split to select the head and detection thresholds. It writes `landmarks.pt` alongside `adapter.pt` so inference loads them together.

## Check camera robustness and fine-tune the heads

Run from `manipulation/panthera_sim` with both trained heads and the landmark labels present. The benchmark pairs each clean scene with four versions of the same RGB frame. It varies exposure, white balance and uneven illumination together, tests defocus or directional blur separately, tests sensor noise with JPEG compression, and combines all three effects. The masks, stud coordinates and camera poses stay aligned.

```bash
.venv-dino/bin/python -m perception.evaluate_camera_robustness \
    --checkpoint outputs/megabloks_dinov3/adapter.pt --split val \
    --output outputs/dino_camera_robustness/baseline_val.json
```

The report includes body and stud mask IoU, block detection by color, and stud localization on the correctly colored block. A block match requires box IoU of at least 0.5, and a stud match requires a center within five pixels of its label. Metrics are also broken down by camera. Paired confidence intervals resample whole scenes.

To train on these appearance changes, start from the original heads and choose a new output directory. The trainer recomputes frozen DINO features from each augmented training image. It selects the two heads independently on clean and augmented validation images.

```bash
.venv-dino/bin/python perception/finetune_camera.py --data data/megabloks \
    --source outputs/megabloks_dinov3 --output outputs/megabloks_augmented
.venv-dino/bin/python -m perception.evaluate_camera_robustness \
    --checkpoint outputs/megabloks_augmented/adapter.pt --split val \
    --reference outputs/dino_camera_robustness/baseline_val.json \
    --output outputs/dino_camera_robustness/candidate_val.json
```

After selecting the checkpoints, run the paired evaluation on `--split test` with separate output files and the test baseline as the reference. For a closed-loop check, run `python -m tools.evaluate_augmented_stack --checkpoint CHECKPOINT --output DIRECTORY` in the same environment. It replays the same layouts under clean and mixed camera conditions. For an ordinary stacking trial, pass the selected `adapter.pt` to `tools.dino_pick_place --checkpoint` and keep the matching `landmarks.pt` beside it.

## Run the trained adapters

Render a new block layout from the tower and run inference on the resulting RGB frame. You can also pass a saved wrist image to `--image`.

```bash
MUJOCO_GL=egl .venv-dino/bin/python -m sim.mega_bloks --seed 0 \
    --output outputs/megabloks_bench/scene.xml --preview outputs/megabloks_bench/tower.png
.venv-dino/bin/python perception/infer.py \
    --checkpoint outputs/megabloks_dinov3/adapter.pt \
    --image outputs/megabloks_bench/tower.png \
    --output outputs/megabloks_dinov3/bench_prediction
```

Open `combined.png` to see orange corner dots, cyan edges, green stud-center crosses and yellow block boxes. In `prediction.json`, the `landmarks` field gives each detected corner and stud center an `[x, y]` position in original-image pixels, a confidence score and a block ID when the assignment is unambiguous. Boxes use `[left, top, right, bottom]` with exclusive right and bottom edges.

Use `edges.png` for the binary edge mask, `semantic.png` for classes and `instance_ids.png` to look up a block ID by pixel. The detector separates touching regions using the six modeled plastic colors. Touching blocks of the same color can merge, and an occluder can split a block into multiple regions.

## Train an adapter for another segmentation task

Use a dataset directory containing `manifest.json` with a `classes` array such as `["background", "gripper", "workpiece"]` and a `splits` object with `train`, `val` and `test` arrays. Each array contains entries such as `{"image": "train/001_rgb.png", "mask": "train/001_mask.png"}`. Group images of the same scene into a single split. Give each image a matching single-channel mask whose integer pixel values index the class array. Reserve ID 0 for background and use a common image height and width, both divisible by 16.

Pass the dataset to `perception/train.py --data` and give the task its own `--output` directory. The trainer sizes the decoder from the class list and saves those names in the checkpoint. Use the same inference command with that checkpoint to produce a semantic mask for the new task.
