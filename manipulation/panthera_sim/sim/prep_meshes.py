"""Prepare Panthera STL meshes for MuJoCo.

The vendor meshes come straight out of SolidWorks: unmerged vertices and up to
321k faces, which is over MuJoCo's 200k-face STL limit and far more detail than
a simulator needs. This writes two reduced sets next to the originals:

  meshes/visual/    decimated, for rendering only
  meshes/collision/ convex hull, for contact

The hull is not a loss of fidelity: MuJoCo replaces any mesh collision geom with
its convex hull anyway, so computing it here just makes that explicit and cheap.
Concave features are therefore not represented: the fingers collide as solid
blocks rather than as hooks, which is fine for reaching and pinching but means
this model will not simulate a finger hooking through a handle.
"""
import argparse
import pathlib
import sys

import trimesh

MESH_DIR = pathlib.Path(__file__).parent / "panthera" / "meshes"


def prep(target_faces: int, hull_faces: int) -> None:
    out_vis = MESH_DIR / "visual"
    out_col = MESH_DIR / "collision"
    out_vis.mkdir(parents=True, exist_ok=True)
    out_col.mkdir(parents=True, exist_ok=True)

    stls = sorted(MESH_DIR.glob("*.STL"))
    if not stls:
        sys.exit(f"no STL files in {MESH_DIR}")

    for path in stls:
        # process=True merges the duplicated vertices SolidWorks emits.
        mesh = trimesh.load(path, process=True)
        n0 = len(mesh.faces)

        vis = mesh
        if n0 > target_faces:
            vis = mesh.simplify_quadric_decimation(face_count=target_faces)
        vis.export(out_vis / path.name)

        hull = mesh.convex_hull
        if len(hull.faces) > hull_faces:
            hull = hull.simplify_quadric_decimation(face_count=hull_faces)
        hull.export(out_col / path.name)

        print(
            f"{path.name:22s} {n0:7d} -> visual {len(vis.faces):6d} "
            f"collision {len(hull.faces):5d}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-faces", type=int, default=40000,
                    help="max faces per visual mesh (default: 40000)")
    ap.add_argument("--hull-faces", type=int, default=256,
                    help="max faces per collision hull (default: 256)")
    a = ap.parse_args()
    prep(a.target_faces, a.hull_faces)
