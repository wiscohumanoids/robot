"""Preview raw episodes: contact sheet PNG (ego camera, 6 frames per episode) + animated GIF of the first episode.
    python preview.py <raw_dir> [--n 6] [--out preview]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = Path(a.raw)
    eps = sorted(raw.glob("episode_*.npz"))[: a.n]
    out = Path(a.out) if a.out else raw.parent / "preview"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in eps:
        z = np.load(e)
        meta = json.loads(e.with_suffix(".json").read_text())
        f = z["ego_rgb"]
        idx = np.linspace(0, len(f) - 1, 6).astype(int)
        tiles = []
        for i in idx:
            im = Image.fromarray(f[i]); ImageDraw.Draw(im).text((4, 4), f"{e.stem[-6:]} t{i} {meta['source']}", fill=(255, 255, 0))
            tiles.append(np.array(im))
        rows.append(np.hstack(tiles))
        print(e.name, meta["source"], meta["active_side"], meta["object_name"], "->", meta["target_name"], f"{len(f)} steps", meta["instruction"])
    Image.fromarray(np.vstack(rows)).save(out / "contact_sheet.png")
    z = np.load(eps[0])
    frames = [Image.fromarray(x) for x in z["ego_rgb"][::2]]
    frames[0].save(out / "episode0_ego.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    print("wrote", out / "contact_sheet.png", out / "episode0_ego.gif")


if __name__ == "__main__":
    main()
