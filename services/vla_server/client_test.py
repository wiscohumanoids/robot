"""Check a running VLA server: GET /health, then POST /act with a test image and zero state.
  python client_test.py [--url http://127.0.0.1:8200] [--image some.png] [--n 5]
Standard library + Pillow."""
import argparse
import base64
import json
import time
import urllib.request



def grey_jpeg():
    """A grey 320x240 JPEG as base64."""
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (320, 240), (128, 110, 90)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def call(url, payload=None):
    """GET (or POST JSON) and return the decoded reply."""
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    """Check a running VLA server: health, then a few /act calls with their latency."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--image")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    h = call(a.url + "/health")
    print("health:", json.dumps(h))
    img = base64.b64encode(open(a.image, "rb").read()).decode() if a.image else grey_jpeg()
    dims = h.get("state") or {"left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1, "waist": 3}
    req = {"instruction": "pick up the cup and place it on the plate", "images": {k: img for k in h["video"]},
           "state": {k: [0.0] * d for k, d in dims.items()}}
    for i in range(a.n):
        t = time.time()
        r = call(a.url + "/act", req)
        if "error" in r:
            raise SystemExit(f"server error: {r['error']}")
        shapes = {k: (len(v), len(v[0])) for k, v in r["actions"].items()}
        print(f"act {i}: server {r['latency_ms']} ms, round trip {1000 * (time.time() - t):.0f} ms, chunk shapes {shapes}")


if __name__ == "__main__":
    main()
