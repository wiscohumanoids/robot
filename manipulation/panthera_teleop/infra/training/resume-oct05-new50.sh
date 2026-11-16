#!/bin/bash
set -euo pipefail
export HF_HOME=/mnt/pi05/cache/huggingface
export TMPDIR=/mnt/pi05/tmp
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PI05_KEEP_CHECKPOINTS=1
export OMP_NUM_THREADS=4
cd /mnt/pi05/lerobot
export PI05_TRAINING_MANIFEST_SHA256=1447c30c3a14f5f705944178d640bcb4c713179fa1250f0d8867702a63dd2967
exec .venv/bin/python -u /mnt/pi05/train_panthera_new50.py \
 --config_path=/mnt/pi05/resume-sources/new50_20261005/024000/pretrained_model/train_config.json \
 --resume=true \
 --dataset.repo_id=FoxNerdSaysMoo/panthera-gear-carrier-pin-oct05-original-camera-20hz-50 \
 --dataset.root=/mnt/pi05/dataset_oct05_original_camera_50 \
 --dataset.revision=null \
 '--dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]' \
 --dataset.eval_split=0 \
 --wandb.run_id=q4pmwlw3 \
 --wandb.resume=must \
 '--wandb.notes=Resumed at step 24000 on 2026-10-05 using ONLY the 50 new slower original-camera demos; prior weights, processors, optimizer and LR schedule preserved.'
