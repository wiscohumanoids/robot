"""VLA inference server (GR00T) for the hsim stack. Runs on the GPU VM inside a GR00T env.

HTTP on 127.0.0.1:8200 (reach it from the Mac with `ssh -N -L 8200:localhost:8200 hsim-gpu`;
the ROS container then uses http://host.docker.internal:8200).

  GET  /health  -> {"ok": true, "model": ..., "state": {key: dim}, "action": {key: dim}, "horizon": H,
                    "execution_horizon": E, "video": [...], "language_key": ..., "mock": bool}
  POST /act     <- {"instruction": str,
                    "images": {"ego_view": "<base64 JPEG or PNG, RGB>"},
                    "state": {"left_arm": [7], "right_arm": [7], "left_gripper": [1], "right_gripper": [1], "waist": [3]}}
                -> {"actions": {"left_arm": [[7] x H], ...}, "horizon": H, "execution_horizon": E, "latency_ms": ms}
Actions are ABSOLUTE joint targets (GR00T converts its relative arm actions back using the state you send);
grippers 0 = closed, 1 = open. Execute the first `execution_horizon` steps at 20 Hz, then ask again.

  python server.py --model-path ~/hsim_runs/M1_finetune_v1/checkpoint-20000
  python server.py --mock            # no GPU / model: returns "hold the current state" chunks (client testing)
"""
import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# mock layout = our g1_dex1 dataset layout (training/manipulation/groot/g1_dex1_config.py)
MOCK_STATE = {"left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1, "waist": 3}
MOCK_ACTION = {"left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1}


def decode_image(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8)


class Backend:
    def __init__(self, args):
        self.lock = threading.Lock()
        self.mock = args.mock
        self.exec_h = args.execution_horizon
        if self.mock:
            self.state_dims, self.action_dims = MOCK_STATE, MOCK_ACTION
            self.video_keys, self.lang_key, self.horizon = ["ego_view"], "annotation.human.task_description", 16
            self.model = "mock"
            return
        import importlib.util
        if args.modality_config:                      # registers NEW_EMBODIMENT before the checkpoint loads
            spec = importlib.util.spec_from_file_location("hsim_modality_cfg", args.modality_config)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        tag = EmbodimentTag(args.embodiment_tag.lower()) if not hasattr(EmbodimentTag, args.embodiment_tag) \
            else getattr(EmbodimentTag, args.embodiment_tag)
        t0 = time.time()
        self.policy = Gr00tPolicy(embodiment_tag=tag, model_path=args.model_path, device=args.device, strict=True)
        mc = self.policy.modality_configs
        self.video_keys = list(mc["video"].modality_keys)
        self.video_t = len(mc["video"].delta_indices)
        self.state_keys = list(mc["state"].modality_keys)
        self.state_t = len(mc["state"].delta_indices)
        self.action_keys = list(mc["action"].modality_keys)
        self.horizon = len(mc["action"].delta_indices)
        self.lang_key = mc["language"].modality_keys[0]
        self.state_dims, self.action_dims = {}, {}
        self.model = args.model_path
        print(f"[vla] loaded {args.model_path} in {time.time() - t0:.1f}s; video {self.video_keys} state {self.state_keys} "
              f"action {self.action_keys} horizon {self.horizon}", flush=True)

    def health(self):
        return {"ok": True, "model": self.model, "mock": self.mock, "state": self.state_dims or None,
                "action": self.action_dims or None, "horizon": self.horizon, "execution_horizon": self.exec_h,
                "video": self.video_keys, "language_key": self.lang_key}

    def act(self, req):
        state = {k: np.asarray(v, np.float32).reshape(-1) for k, v in req["state"].items()}
        if self.mock:
            acts = {k: np.repeat(state.get(k, np.zeros(d, np.float32))[None], self.horizon, 0) for k, d in MOCK_ACTION.items()}
            return {k: v.tolist() for k, v in acts.items()}
        obs = {"video": {}, "state": {}, "language": {self.lang_key: [[req["instruction"]]]}}
        for k in self.video_keys:
            img = decode_image(req["images"][k])
            obs["video"][k] = np.repeat(img[None, None], self.video_t, 1)          # (1, T, H, W, 3)
        for k in self.state_keys:
            obs["state"][k] = np.repeat(state[k][None, None], self.state_t, 1)    # (1, T, D)
            self.state_dims[k] = int(state[k].shape[0])
        with self.lock:
            action, _ = self.policy.get_action(obs)
        out = {}
        for k in self.action_keys:
            a = np.asarray(action[k])[0]                                             # (H, D)
            self.action_dims[k] = int(a.shape[-1])
            out[k] = a.astype(float).tolist()
        return out


def make_handler(backend):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, backend.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/act":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.time()
                acts = backend.act(req)
                self._send(200, {"actions": acts, "horizon": backend.horizon, "execution_horizon": backend.exec_h,
                                 "latency_ms": round((time.time() - t0) * 1000, 1)})
            except Exception as e:                       # report errors to the client instead of dropping the socket
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path")
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--modality-config", default=None,
                    help="py file that registers the NEW_EMBODIMENT config (needed if the checkpoint lacks it)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--execution-horizon", type=int, default=8)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if not a.mock and not a.model_path:
        ap.error("--model-path is required unless --mock")
    backend = Backend(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(backend))
    print(f"[vla] serving on http://{a.host}:{a.port} (mock={a.mock})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
