"""Convert the Onshape assembly OBJ to local meshes and convex section prisms.

CAD export is Z-up, metres. Collision meshes retain holes and gear teeth;
piecewise constant XY cross-sections approximate curved bevels by slabs.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely import constrained_delaunay_triangles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'sim/actuator'
SOURCE = OUT / 'source/Wisco_Actuator_Gear_Motion.obj'


def read_groups(path):
    vertices, groups = [], []
    for line in path.read_text().splitlines():
        items = line.split()
        if not items:
            continue
        if items[0] == 'v':
            vertices.append([float(x) for x in items[1:4]])
        elif items[0] == 'g':
            groups.append((items[1], []))
        elif items[0] == 'f':
            f = [int(x.split('/')[0])-1 for x in items[1:]]
            for k in range(1, len(f)-1):
                groups[-1][1].append([f[0], f[k], f[k+1]])
    vertices = np.asarray(vertices)
    for name, faces in groups:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fix_normals()
        yield name, mesh


def merge_convex(polygons):
    """Merge adjacent triangles only if their union is still convex."""
    polygons = list(polygons)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(polygons):
            for j in range(i+1, len(polygons)):
                b = polygons[j]
                if not a.intersects(b):
                    continue
                u = a.union(b)
                if isinstance(u, Polygon) and abs(u.convex_hull.area-u.area) < 1e-14:
                    polygons[i] = u.convex_hull
                    polygons.pop(j)
                    changed = True
                    break
            if changed:
                break
    return polygons


def collision_sections(mesh):
    # Axial step heights come directly from planar CAD faces, avoiding tiny
    # slabs from tessellated fillets. Every retained horizontal face is a cut.
    mask = np.abs(mesh.face_normals[:, 2]) > .999999
    levels = np.unique(np.round(mesh.triangles_center[mask, 2], 7))
    levels = np.unique(np.r_[mesh.bounds[:, 2], levels])
    pieces = []
    for low, high in zip(levels[:-1], levels[1:]):
        if high-low < 1e-6:
            continue
        section = mesh.section([0, 0, 1], [0, 0, (low+high)/2])
        if section is None:
            continue
        # to_2D changes the frame; transform outlines back to world XY.
        planar, transform = section.to_2D()
        for poly in planar.polygons_full:
            def xy(coords):
                v = np.c_[np.asarray(coords), np.zeros(len(coords)), np.ones(len(coords))]
                return (v @ transform.T)[:, :2]
            poly = Polygon(xy(poly.exterior.coords), [xy(r.coords) for r in poly.interiors])
            poly = poly.simplify(0.000015, preserve_topology=True)
            tris = list(constrained_delaunay_triangles(poly).geoms)
            for p in merge_convex(tris):
                points = np.asarray(p.exterior.coords)[:-1]
                verts = np.vstack([np.c_[points, np.full(len(points), z)] for z in (low, high)])
                pieces.append(trimesh.convex.convex_hull(verts))
    return pieces


def main():
    manifest = dict(source_url='https://cad.onshape.com/documents/6ebda61ccb55656f56193289/w/a3ae93aac555a13ff6d266af/e/b429d4c9da58a01075feb849',
                    export_date='2026-09-28', units='metres', up_axis='Z',
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    collision_method='horizontal CAD sections, 15 micrometre outline simplification, convex prisms', parts={})
    pins = 0
    for source_name, mesh in read_groups(SOURCE):
        if source_name.startswith(('01_', '02_')):
            continue
        if source_name.startswith('06_'):
            pins += 1
            name = f'pin_{pins}'
        elif source_name.startswith('05_'):
            name = 'gear_' + source_name[-1]
        elif source_name.startswith('03_'):
            name = 'small_carrier'
        else:
            name = 'large_carrier'
        origin = np.r_[mesh.bounds.mean(axis=0)[:2], mesh.bounds[0, 2]]
        if 'carrier' in name:
            origin[:2] = 0
        mesh.apply_translation(-origin)
        folder = OUT / 'meshes' / name
        folder.mkdir(parents=True, exist_ok=True)
        mesh.export(folder / 'visual.stl')
        pieces = collision_sections(mesh)
        for i, piece in enumerate(pieces):
            piece.export(folder / f'collision_{i:03d}.stl')
        manifest['parts'][name] = dict(source_name=source_name, assembled_origin=origin.tolist(),
            bounds=mesh.bounds.tolist(), volume_m3=float(abs(mesh.volume)), collision_pieces=len(pieces))
        print(name, origin, mesh.extents, len(pieces), flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
