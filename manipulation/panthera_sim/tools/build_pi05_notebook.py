#!/usr/bin/env python3
"""Generate the self-contained RTX PRO 6000 notebook (no local checkout needed)."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})


def code(source):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": textwrap.dedent(source).strip() + "\n"})


def main():
    CELLS.clear()
    release_path = ROOT / "outputs/pi05_artifacts/release.json"
    release = json.loads(release_path.read_text()) if release_path.exists() else {"revision": "main"}
    md('''
    # Full-model π₀.₅: Panthera IK three-block stacking

    Target: **one NVIDIA RTX PRO 6000, 96 GB VRAM**, Linux, 64 GB+ system RAM
    (128 GB recommended). Storage depends on checkpoints and caches; budget roughly
    **100–150 GiB total**, not 300 GiB free at startup. The notebook prints free space,
    retains the latest two full checkpoints, and measures actual checkpoint sizes.
    The smoke run saves no model/optimizer copy. Use a current Blackwell-compatible
    NVIDIA driver; the pinned LeRobot environment uses CUDA 12.8 PyTorch wheels.
    A 48 GB RTX 6000 Ada is a different GPU. The VRAM preflight catches this.

    This trains **all model parameters**: vision tower/projector, language backbone,
    action expert, and action projections. No LoRA, quantization, or frozen backbone.
    It verifies optimizer coverage and nonzero backward gradients in every major
    component. Activation checkpointing and bfloat16 reduce memory, without freezing
    layers. Batch size 8 is a starting point, not a measured memory guarantee.

    Dataset: **1,000 scripted IK demos / 862,766 frames**, 30 Hz, shoulder + wrist RGB.
    Task: red onto green, then blue onto red. Inputs are six measured joint angles
    plus the gripper opening command; labels are **absolute next-step joint targets
    and gripper opening in metres**. The export already shifts actions once.
    Keep these joint-space labels even though IK generated the demonstrations.

    Run in order. The four-update smoke run tests the complete training path before
    the 30,000-update run. Then use **Watch rollouts** for three videos and **Measure
    success** for 50 fresh seeds per start distribution. Those cells can be rerun
    independently on any saved checkpoint. Offline validation loss is not task success.

    [Dataset](https://huggingface.co/datasets/FoxNerdSaysMoo/panthera-ik-three-block-stack-30hz)
    · [LeRobot pi0.5 documentation](https://huggingface.co/docs/lerobot/pi05)
    · [Pinned training implementation](https://github.com/huggingface/lerobot/tree/e624f3f7f8411ec3a02635d06e79373341e5ef35)
    ''')
    code('''
    from pathlib import Path
    # Set WORK_DIR to a persistent SSD/NVMe mount on your GPU machine.
    WORK_DIR = (Path.cwd() / "pi05_ik3").resolve()
    RUN_NAME = "pi05_ik3_full_001"  # Change for a new experiment.
    SAVE_TO_DRIVE = True            # Mount/authenticate before setup; back up every completed checkpoint.
    DRIVE_ROOT = Path("/content/drive/MyDrive/pi05_ik3")
    RESUME_FROM_DRIVE = False       # Restore the latest backup for RUN_NAME, including optimizer/RNG.
    DATASET_REPO = "FoxNerdSaysMoo/panthera-ik-three-block-stack-30hz"
    DATASET_REVISION = "DATASET_REVISION_PLACEHOLDER"
    LEROBOT_COMMIT = "e624f3f7f8411ec3a02635d06e79373341e5ef35"
    BASE_MODEL = "lerobot/pi05_base"
    BASE_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
    BATCH_SIZE = 8
    TRAIN_STEPS = 30_000             # Optimizer updates; accumulation is 1.
    SAVE_FREQ = 5_000                # Each checkpoint includes optimizer/RNG state.
    EVAL_LOSS_FREQ = 1_000
    LEARNING_RATE = 2.5e-5
    CHUNK_SIZE = 50                  # Predict 1.67 s of actions at 30 Hz.
    ACTION_STEPS = 10                # Replan every 0.33 s during rollout.
    NUM_WORKERS = 8
    SEED = 20260925
    KEEP_CHECKPOINTS = 2             # Latest two complete checkpoints in THIS run.
                                    # Set 1 for less disk use; saves replacement before pruning.
    STORAGE_WARNING_GIB = 120        # Advisory only; does not block setup.
    RESUME_CHECKPOINT = ""          # .../checkpoints/last/pretrained_model
    CHECKPOINT_OVERRIDE = ""        # Evaluate a different saved pretrained_model directory.
    EVAL_EPISODES = 50               # Per distribution; total 100 for both.
    EVAL_SECONDS = 60                # Demos last up to 41 s; allow recovery time.
    EVAL_SEED = 260925000            # Fresh fixed seeds, disjoint from collection.
    VIDEO_EPISODES = 6               # Per distribution; -1 records every rollout.
    WANDB_ENABLE = True
    WANDB_PROJECT = "panthera-pi05-ik3"
    WANDB_ENTITY = ""                # Optional user/team; empty uses your default.
    WANDB_MODE = "online"            # "offline" keeps logs locally for later sync.
    WANDB_LOG_VIDEOS = True          # Upload saved rollout videos; never model weights.
    ''')
    md('''
    ## 1. Install an isolated, pinned environment

    **Google Drive access is requested first**, before package installation or
    downloads. Approve it now so checkpoints can be backed up unattended later.
    Only the latest complete checkpoint for this run is retained on Drive; a new
    copy finishes before the old one is removed. Both model and optimizer/RNG state
    are included for resuming. Data, base weights, and caches stay on local disk.

    The Jupyter kernel only orchestrates subprocesses; training uses its own Python
    3.12 environment. No kernel restart is needed. Downloads and logs stream into
    the cells. Setup does not modify your other Python environments.
    ''')
    code('''
    import os, sys, subprocess, shutil, json, time
    if RESUME_FROM_DRIVE and not SAVE_TO_DRIVE:
        raise ValueError("RESUME_FROM_DRIVE requires SAVE_TO_DRIVE=True")
    DRIVE_RUN_DIR = DRIVE_ROOT / RUN_NAME
    if SAVE_TO_DRIVE:
        try:
            from google.colab import drive
        except ImportError:
            if not DRIVE_ROOT.is_dir():
                raise RuntimeError("Outside Colab, mount Drive yourself and set DRIVE_ROOT to that existing folder, "
                                   "or set SAVE_TO_DRIVE=False.")
        else:
            drive.mount("/content/drive")  # Any access prompt happens BEFORE installation/downloads.
        DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
        import tempfile
        with tempfile.NamedTemporaryFile(dir=DRIVE_RUN_DIR, prefix=".pi05-write-probe-") as probe:
            probe.write(b"pi05 checkpoint backup write check"); probe.flush()
        print("Drive checkpoint backup ready:", DRIVE_RUN_DIR)
        if RESUME_FROM_DRIVE and not (DRIVE_RUN_DIR / "latest.json").is_file():
            raise FileNotFoundError(f"No saved checkpoint pointer in {DRIVE_RUN_DIR}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(WORK_DIR).free / 2**30
    print(f"Free storage on {WORK_DIR}: {free_gib:.1f} GiB")
    if free_gib < STORAGE_WARNING_GIB:
        print("Storage advisory: data is ~22 GiB, plus base weights, environment/caches, and checkpoints. "
              "Reusing existing downloads needs less additional space. "
              "KEEP_CHECKPOINTS=1 reduces retention; space is checked against actual tensors before each save.")
    if KEEP_CHECKPOINTS < 1:
        raise ValueError("KEEP_CHECKPOINTS must be at least 1")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("No NVIDIA driver found; run this on the RTX PRO 6000 server.")

    def run(argv, *, cwd=None, env=None, log=None):
        argv = list(map(str, argv))
        print("+", " ".join(argv), flush=True)
        output = open(log, "a") if log else None
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                if output:
                    output.write(line); output.flush()
            if proc.wait():
                raise subprocess.CalledProcessError(proc.returncode, argv)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
        finally:
            if output:
                output.close()

    run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
    run([sys.executable, "-m", "pip", "install", "-q", "uv", "huggingface_hub", "ipywidgets"])
    UV = [sys.executable, "-m", "uv"]
    LEROBOT_DIR = WORK_DIR / "lerobot"
    if not LEROBOT_DIR.exists():
        run(["git", "init", LEROBOT_DIR])
        run(["git", "remote", "add", "origin", "https://github.com/huggingface/lerobot.git"], cwd=LEROBOT_DIR)
    run(["git", "fetch", "--depth=1", "origin", LEROBOT_COMMIT], cwd=LEROBOT_DIR)
    run(["git", "checkout", "--detach", LEROBOT_COMMIT], cwd=LEROBOT_DIR)
    CHILD_ENV = dict(os.environ, UV_CACHE_DIR=str(WORK_DIR / "uv-cache"),
                     TMPDIR=str(WORK_DIR / "tmp"), TOKENIZERS_PARALLELISM="false", MUJOCO_GL="egl",
                     CUDA_VISIBLE_DEVICES="0", PI05_KEEP_CHECKPOINTS=str(KEEP_CHECKPOINTS))
    Path(CHILD_ENV["TMPDIR"]).mkdir(exist_ok=True)
    if SAVE_TO_DRIVE:
        CHILD_ENV["PI05_DRIVE_BACKUP_ROOT"] = str(DRIVE_RUN_DIR)
    else:
        CHILD_ENV.pop("PI05_DRIVE_BACKUP_ROOT", None)
    run(UV + ["sync", "--frozen", "--extra", "pi", "--extra", "training", "--python", "3.12"],
        cwd=LEROBOT_DIR, env=CHILD_ENV)
    PYTHON = LEROBOT_DIR / ".venv/bin/python"
    run(UV + ["pip", "install", "--python", PYTHON, "mujoco==3.13.0", "imageio==2.37.4", "imageio-ffmpeg==0.6.0"],
        env=CHILD_ENV)
    def python(source, *, log=None):
        run([PYTHON, "-u", "-c", source], env=CHILD_ENV, log=log)
    python("import torch; assert torch.cuda.is_available(); "
           "p=torch.cuda.get_device_properties(0); print(p); "
           "assert p.total_memory/2**30 >= 85, 'Select the 96 GB RTX PRO 6000'; "
           "assert torch.cuda.is_bf16_supported(); "
           "x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16); "
           "print('CUDA bf16 matmul OK:', (x@x).shape, 'torch:',torch.__version__)")
    ''')
    md('''
    ## 2. Hugging Face access and reproducible downloads

    Accept the [PaliGemma tokenizer license](https://huggingface.co/google/paligemma-3b-pt-224)
    once using the account behind your read token. The token is entered securely or
    reused from the environment; it is not stored in notebook cells or outputs.
    The training dataset is about 22 GiB. The runtime bundle contains the exact
    simulator scene/meshes, camera definitions, and training/evaluation scripts.
    ''')
    code('''
    from huggingface_hub import HfApi, get_token
    from getpass import getpass
    if WANDB_MODE not in ("online", "offline"):
        raise ValueError("WANDB_MODE must be online or offline")
    if WANDB_ENABLE and WANDB_MODE == "online":
        # The child process sees the key through its environment, not CLI/logs.
        wandb_key = os.environ.get("WANDB_API_KEY") or getpass("W&B API key (wandb.ai/authorize): ")
        if not wandb_key:
            raise RuntimeError("Provide a W&B API key, or select WANDB_MODE='offline'.")
        CHILD_ENV["WANDB_API_KEY"] = wandb_key
        del wandb_key
    token = os.environ.get("HF_TOKEN") or get_token() or getpass("Hugging Face read token: ")
    if not token:
        raise RuntimeError("A token with PaliGemma tokenizer access is required.")
    CHILD_ENV.update(HF_TOKEN=token, HF_HOME=str(WORK_DIR / "hf-cache"),
                     HF_DATASETS_CACHE=str(WORK_DIR / "hf-cache/datasets"),
                     HF_LEROBOT_HOME=str(WORK_DIR / "hf-cache/lerobot"))
    del token
    DATA_ROOT = WORK_DIR / "dataset"
    RUNTIME = WORK_DIR / "runtime"
    MODEL_ROOT = WORK_DIR / "pi05_base"
    download_settings = dict(repo=DATASET_REPO, revision=DATASET_REVISION, data=str(DATA_ROOT),
                             runtime=str(RUNTIME), model=BASE_MODEL, model_revision=BASE_REVISION,
                             model_root=str(MODEL_ROOT))
    CHILD_ENV["PI05_DOWNLOAD_SETTINGS"] = json.dumps(download_settings)
    python(r"""
    import hashlib, json, os, zipfile
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    s=json.loads(os.environ['PI05_DOWNLOAD_SETTINGS'])
    AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')
    snapshot_download(s['repo'], repo_type='dataset', revision=s['revision'], local_dir=s['data'],
        allow_patterns=['data/**','meta/**','COMPLETE','verification.json','upload_manifest.json','assets/pi05_runtime.zip'])
    root=Path(s['data'])
    assert (root/'COMPLETE').is_file(), 'Dataset publication is incomplete'
    info=json.loads((root/'meta/info.json').read_text())
    assert (info['fps'],info['total_episodes'],info['total_frames']) == (30,1000,862766)
    for key in ('observation.state','action'):
        assert info['features'][key]['shape']==[7]
    for key in ('observation.images.shoulder','observation.images.wrist'):
        assert info['features'][key]['shape']==[256,256,3]
    # Validate the published hashes without copying the Parquet files.
    manifest=json.loads((root/'upload_manifest.json').read_text())
    for name,entry in manifest['files'].items():
        if not (name.startswith('data/') or name.startswith('meta/')): continue
        p=root/name; h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
        assert p.stat().st_size==entry['size'] and h.hexdigest()==entry['sha256'], name
    dest=Path(s['runtime']); dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(root/'assets/pi05_runtime.zip') as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(dest)
    runtime=json.loads((dest/'runtime_manifest.json').read_text())
    for name,sha in runtime['files'].items():
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==sha, name
    snapshot_download(s['model'],revision=s['model_revision'],local_dir=s['model_root'],
        allow_patterns=['config.json','model.safetensors'])
    print('Dataset, tokenizer, runtime hashes, and pretrained checkpoint downloaded.')
    """)
    CHILD_ENV["PYTHONPATH"] = str(RUNTIME)
    training_seeds = set(json.loads((RUNTIME / "runtime_manifest.json").read_text())["training_seeds"])
    assert not training_seeds.intersection(range(EVAL_SEED, EVAL_SEED + EVAL_EPISODES))
    ''')
    md('''
    ## 3. Inspect the two training cameras and test headless rendering

    Training uses both cameras in this order. π₀.₅ resizes 256×256 RGB to 224×224
    internally. The state gripper value is a command, matching deployment. Quantile
    normalization is recomputed on episodes **0–899 only**; episodes **900–999**
    are held out for offline validation. There is no frame-level split leakage.
    ''')
    code('''
    CHILD_ENV["PI05_DATA_ROOT"] = str(DATA_ROOT)
    CHILD_ENV["PI05_WORK_DIR"] = str(WORK_DIR)
    python(r"""
    import io, os
    from pathlib import Path
    import pyarrow.parquet as pq
    from PIL import Image
    import mujoco
    from sim.panthera_env import PantheraSim
    from teleop.render_vla_dataset import shoulder_camera, wrist_camera
    p=next((Path(os.environ['PI05_DATA_ROOT'])/'data').rglob('*.parquet'))
    row=next(pq.ParquetFile(p).iter_batches(batch_size=1)).to_pylist()[0]
    canvas=Image.new('RGB',(512,256))
    for i,key in enumerate(('observation.images.shoulder','observation.images.wrist')):
        canvas.paste(Image.open(io.BytesIO(row[key]['bytes'])), (256*i,0))
    canvas.save(Path(os.environ['PI05_WORK_DIR'])/'dataset_cameras.png')
    sim=PantheraSim()
    for camera in (shoulder_camera(sim.model),wrist_camera(sim.model)):
        with mujoco.Renderer(sim.model,height=256,width=256) as renderer:
            renderer.update_scene(sim.data,camera); assert renderer.render().shape==(256,256,3)
    print('Both dataset cameras decoded; EGL simulator rendering works.')
    """)
    from IPython.display import display, Image as NotebookImage
    display(NotebookImage(filename=str(WORK_DIR / "dataset_cameras.png")))
    ''')
    md('''
    ## 4. Full-model configuration and four-update smoke run

    Each update includes every trainable weight. `full_model_audit.json` checks
    that none is frozen, every parameter belongs to AdamW, and gradients reach
    the vision encoder/projector, language layers, action expert and action output.
    Unused language-generation heads can have no gradient because this is an action
    loss; no layer is intentionally frozen. Weight-loading failures are fatal.

    The smoke run runs held-out loss and saves only small diagnostic files. Inspect
    its peak VRAM before continuing. If it runs out of memory, reduce **BATCH_SIZE**
    and rerun with a fresh smoke directory; it never switches to LoRA or freezes
    the vision encoder. Batch size 16 is an optional later throughput experiment.
    ''')
    code('''
    TRAINER = RUNTIME / "tools/train_pi05_full.py"
    OUTPUT = WORK_DIR / "runs" / RUN_NAME
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR = WORK_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    def wandb_flags():
        flags = [f"--wandb.enable={str(WANDB_ENABLE).lower()}", f"--wandb.project={WANDB_PROJECT}",
                 f"--wandb.mode={WANDB_MODE}", "--wandb.disable_artifact=true"]
        if WANDB_ENTITY:
            flags.append(f"--wandb.entity={WANDB_ENTITY}")
        return flags
    def train_command(output, steps, save_freq, eval_freq):
        return [PYTHON, "-u", TRAINER,
            f"--dataset.repo_id={DATASET_REPO}", f"--dataset.root={DATA_ROOT}",
            f"--dataset.revision={DATASET_REVISION}", "--dataset.eval_split=0.1",
            "--dataset.use_imagenet_stats=false", "--dataset.video_backend=pyav",
            "--policy.type=pi05", f"--policy.pretrained_path={MODEL_ROOT}",
            "--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false",
            "--policy.use_relative_actions=false", "--policy.empty_cameras=0",
            "--policy.gradient_checkpointing=true", "--policy.compile_model=false",
            "--policy.dtype=bfloat16", "--policy.device=cuda", "--policy.push_to_hub=false",
            '--policy.normalization_mapping={"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}',
            f"--policy.chunk_size={CHUNK_SIZE}", f"--policy.n_action_steps={ACTION_STEPS}",
            f"--policy.optimizer_lr={LEARNING_RATE}",
            f"--policy.scheduler_decay_steps={TRAIN_STEPS}",
            f"--policy.scheduler_warmup_steps={min(1000,max(1,steps//10))}",
            "--accelerator.gradient_accumulation.steps=1", "--accelerator.mixed_precision=bf16",
            f"--output_dir={output}", f"--job_name={RUN_NAME}", f"--batch_size={BATCH_SIZE}",
            f"--num_workers={NUM_WORKERS}", "--prefetch_factor=2",
            f"--steps={steps}", f"--save_freq={save_freq}", f"--eval_steps={eval_freq}",
            "--max_eval_samples=128", "--env_eval_freq=0", "--log_freq=20",
            f"--seed={SEED}"] + wandb_flags()
    SMOKE_OUTPUT = WORK_DIR / "runs" / (RUN_NAME + "_smoke_" + time.strftime("%Y%m%d_%H%M%S"))
    if not RESUME_CHECKPOINT and not RESUME_FROM_DRIVE:
        smoke_command = [a for a in train_command(SMOKE_OUTPUT, 4, 4, 4) if not str(a).startswith("--wandb.")]
        run(smoke_command + ["--save_checkpoint=false", "--wandb.enable=false"],
            env=CHILD_ENV, log=LOG_DIR / "smoke.log")
        print((SMOKE_OUTPUT / "full_model_audit.json").read_text())
        print((SMOKE_OUTPUT / "peak_memory.json").read_text())
    else:
        print("Resume selected; the training cell reruns full-model guards on the restored checkpoint.")
    ''')
    md('''
    ## 5. Train or resume

    Fresh training starts from π₀.₅ base again, so smoke updates do not alter the
    experiment. Checkpoints retain model, preprocessing statistics, optimizer,
    scheduler, and RNG state. To resume, set `RESUME_CHECKPOINT` above to the saved
    `checkpoints/last/pretrained_model` directory and rerun setup/downloads, the
    configuration cell, and this cell. The checkpoint's training configuration is
    authoritative; `TRAIN_STEPS` is the desired **total**, not extra steps.

    To resume after a Colab reset, keep the same `RUN_NAME`/`DRIVE_ROOT` and set
    **`RESUME_FROM_DRIVE=True`**. The latest Drive backup is copied back to local
    storage and its checksums verified before training resumes. Each checkpoint
    save backs up the complete resumable checkpoint to
    `MyDrive/pi05_ik3/<RUN_NAME>/<step>/`; `latest.json` identifies the latest copy.
    Drive needs space for the current backup plus its replacement while copying.
    Backup failures stop with an error while preserving the local checkpoint and
    the previous complete Drive backup. No Drive access prompt occurs at save time.

    Use persistent storage. By default only the latest two completed checkpoints
    from this run are retained. A replacement is fully saved before an older one
    is removed, so temporary space for a third is needed (or two if retention is 1).
    Past experiments and checkpoints predating this retention mechanism are not
    deleted. `storage.json` reports actual checkpoint size and remaining free disk.
    Training and rollouts run sequentially to avoid competing for VRAM.

    W&B is enabled by default: training/validation loss, learning rate, gradient
    norm, timing, and system metrics go to `WANDB_PROJECT`. Its run URL appears in
    the training log. Resume uses the saved W&B run ID. Model artifacts are disabled
    to avoid uploading multi-GB checkpoints. Offline mode can be synced later with
    `wandb sync <offline-run-directory>` from the isolated environment.
    ''')
    code('''
    if RESUME_FROM_DRIVE:
        CHILD_ENV["PI05_RESTORE_ROOT"] = str(WORK_DIR / "restored_checkpoints" / RUN_NAME)
        python(r"""
    import json, os
    from pathlib import Path
    from tools.pi05_checkpoint_backup import restore_latest_checkpoint
    restored = restore_latest_checkpoint(os.environ['PI05_DRIVE_BACKUP_ROOT'], os.environ['PI05_RESTORE_ROOT'])
    (Path(os.environ['PI05_RESTORE_ROOT'])/'restored.json').write_text(json.dumps({'path': str(restored)}))
    print('Verified and restored Drive checkpoint:', restored)
    """)
        RESUME_CHECKPOINT = json.loads((Path(CHILD_ENV["PI05_RESTORE_ROOT"]) / "restored.json").read_text())["path"]
    if RESUME_CHECKPOINT:
        resume = Path(RESUME_CHECKPOINT).expanduser().resolve()
        assert (resume / "train_config.json").is_file(), resume
        resume_cfg = json.loads((resume / "train_config.json").read_text())
        OUTPUT = Path(resume_cfg["output_dir"])
        command = [PYTHON, "-u", TRAINER, f"--config_path={resume / 'train_config.json'}",
                   "--resume=true", f"--steps={TRAIN_STEPS}", f"--dataset.root={DATA_ROOT}"] + wandb_flags()
        if WANDB_ENABLE and not resume_cfg.get("wandb", {}).get("run_id"):
            # A checkpoint trained without W&B needs a new logging run, even
            # though model/optimizer training resumes normally.
            import uuid
            command += [f"--wandb.run_id={uuid.uuid4().hex[:8]}", "--wandb.resume=allow"]
    else:
        command = train_command(OUTPUT, TRAIN_STEPS, SAVE_FREQ, EVAL_LOSS_FREQ)
    run(command, env=CHILD_ENV, log=LOG_DIR / (RUN_NAME + ".log"))
    print("Finished. Last checkpoint:", OUTPUT / "checkpoints/last/pretrained_model")
    print((OUTPUT / "full_model_audit.json").read_text())
    print((OUTPUT / "peak_memory.json").read_text())
    ''')
    md('''
    ## 6. Watch rollouts

    These are closed-loop policy rollouts from fresh seeds. Simulation advances at
    exactly 30 Hz in simulated time; inference can run slower than real time.
    The original contact-triggered grasp assistance is unchanged from the demos.
    Video overlays show elapsed simulation time and the stable-stack timer.

    **Success:** green on the table, red on green, blue on red; adjacent horizontal
    error ≤12 mm; vertical gaps 32–60 mm; no active grasp; held for one full second.
    Evaluation stops at success or 60 simulated seconds. No expert rescue, object
    teleports during rollout, or extra joint-step cap is used. Action clipping to
    joint/gripper limits is reported. Errors count as failures and are listed.
    ''')
    code('''
    CHECKPOINT = (Path(CHECKPOINT_OVERRIDE).expanduser() if CHECKPOINT_OVERRIDE
                  else OUTPUT / "checkpoints/last/pretrained_model").resolve()
    assert (CHECKPOINT / "deployment.json").is_file(), CHECKPOINT
    EVALUATOR = RUNTIME / "tools/evaluate_pi05.py"
    def evaluate(output, episodes, distribution, video_episodes):
        wandb_args = []
        if WANDB_ENABLE:
            wandb_args = ["--wandb", "--wandb-project", WANDB_PROJECT, "--wandb-mode", WANDB_MODE]
            if WANDB_ENTITY:
                wandb_args += ["--wandb-entity", WANDB_ENTITY]
            if WANDB_LOG_VIDEOS:
                wandb_args += ["--wandb-log-videos"]
        run([PYTHON, "-u", EVALUATOR, "--checkpoint", CHECKPOINT, "--output", output,
             "--episodes", episodes, "--seconds", EVAL_SECONDS, "--seed", EVAL_SEED,
             "--distribution", distribution, "--video-episodes", video_episodes,
             "--action-steps", ACTION_STEPS] + wandb_args, env=CHILD_ENV, log=LOG_DIR / "rollouts.log")
        return json.loads((output / "summary.json").read_text())
    PREVIEW_DIR = WORK_DIR / "evaluations" / (RUN_NAME + "_preview_" + time.strftime("%Y%m%d_%H%M%S"))
    preview = evaluate(PREVIEW_DIR, 3, "matched", -1)
    from IPython.display import Video, Markdown
    for record in preview["records"]:
        display(Markdown(f"**Seed {record['seed']}: {'SUCCESS' if record['success'] else 'FAILURE'}**"))
        if record["video"]:
            display(Video(record["video"], embed=True, width=768))
        if record["error"]:
            print(record["error"])
    ''')
    md('''
    ## 7. Measure task success on fixed seeds

    `matched` uses the scripted collector's reachable start region/orientation,
    with fresh seeds and **without filtering for planner success**. `broad` uses
    teleoperation's larger 10–40 cm start-height range. Keep their scores separate.
    These seeds are disjoint from collection, and identical across checkpoints.
    Do not tune repeatedly on a final test seed range; reserve another range for
    final reporting. The preview seeds are included in this development benchmark.

    A 50-episode result has sampling uncertainty. The report includes a 95% Wilson
    interval, all per-seed outcomes, grasp/lift/two-stack rates, and timing. Re-run
    this section with `CHECKPOINT_OVERRIDE` to compare checkpoints. Checkpoint
    paths, action execution horizon and metric settings are saved with results.
    With W&B enabled, each preview/benchmark gets an evaluation run grouped by the
    training run ID, with success rates, confidence intervals, intermediate metrics,
    and saved videos. Set `WANDB_LOG_VIDEOS=False` for metrics-only logging.
    ''')
    code('''
    BENCHMARK_DIR = WORK_DIR / "evaluations" / (RUN_NAME + "_benchmark_" + time.strftime("%Y%m%d_%H%M%S"))
    benchmark = evaluate(BENCHMARK_DIR, EVAL_EPISODES, "both", VIDEO_EPISODES)
    rows = ["| Start distribution | Success | Rate | 95% interval | Grasp | Lift | Two-stack | Errors |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, result in benchmark["by_distribution"].items():
        lo, hi = result["success_95pct_wilson_interval"]
        rates = result["rates"]
        rows.append(f"| {name} | {result['successes']}/{result['episodes']} | {result['success_rate']:.1%} | "
                    f"{lo:.1%}–{hi:.1%} | {rates['grasped']:.1%} | {rates['lifted']:.1%} | "
                    f"{rates['two_stacked']:.1%} | {result['error_episodes']} |")
    display(Markdown("\\n".join(rows)))
    print("Complete evaluation report:", BENCHMARK_DIR / "summary.json")
    ''')
    code('''
    # Browse saved successes and failures without rerunning inference.
    import ipywidgets as widgets
    video_records = [r for r in benchmark["records"] if r["video"] and Path(r["video"]).is_file()]
    if video_records:
        options = [(f"{r['distribution']} / seed {r['seed']} / "
                    f"{'SUCCESS' if r['success'] else 'FAILURE'}", i) for i, r in enumerate(video_records)]
        chooser = widgets.Dropdown(options=options, description="Rollout:", layout=widgets.Layout(width="700px"))
        viewer = widgets.Output()
        def show_rollout(change=None):
            r = video_records[chooser.value]
            with viewer:
                viewer.clear_output(wait=True)
                display(Video(r["video"], embed=True, width=768))
                print(json.dumps({k:v for k,v in r.items() if k != "video"}, indent=2))
        chooser.observe(show_rollout, names="value")
        display(chooser, viewer)
        show_rollout()
    else:
        print("No videos recorded. Set VIDEO_EPISODES=-1 to record every rollout; inspect errors in summary.json.")
    ''')
    md('''
    ## Outputs and reuse

    - **Deployable policy:** `runs/<RUN_NAME>/checkpoints/last/pretrained_model`,
      including learned weights, camera/action config, processors, training-only
      normalization statistics, and `deployment.json`.
    - **Resume training:** retain the entire checkpoint directory, including
      `training_state`; the model-only directory is insufficient to resume AdamW.
      With `SAVE_TO_DRIVE=True`, that complete checkpoint is backed up automatically
      under `DRIVE_ROOT/RUN_NAME` after every save. The smoke run has no checkpoint
      to back up. Drive keeps one complete backup, regardless of local retention.
    - **Full-model proof and peak memory:** `full_model_audit.json` and
      `peak_memory.json` in the run directory. These are written by an actual run.
    - **Videos and success rates:** `evaluations/*/summary.json`, per-seed JSON,
      browser-playable H.264 MP4s; optional `--save-traces` stores simulator states.

    No model is published automatically. This notebook prepares and executes a
    full fine-tune; its presence alone is not evidence of trained task success.
    ''')
    for cell in CELLS:
        cell["source"] = cell["source"].replace("DATASET_REVISION_PLACEHOLDER", release["revision"])
    notebook = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"pi05-{i:02d}"
    path = ROOT / "notebooks/pi05_ik_three_block_full_finetune.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
