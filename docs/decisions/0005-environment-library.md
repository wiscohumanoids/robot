# 0005 — Environments come from a library, not only a phone scan

**Context:** The brief's Phase 2 was "phone scan → 3D reconstruction → MuJoCo scene". A scan is an unlabeled mesh: nothing in it says "table" or "mug", which Phases 3–5 need (planner targets, perception ground truth). Curated environments come labeled.

**Decision**
1. Environments are plug-ins. Every source produces the same outputs under `assets/envs/<source>/<name>/build/`:
   - `world.mjb`: G1 training world + the environment, compiled to MuJoCo's self-contained binary format (meshes embedded, MuJoCo 3.11)
   - `map.pgm` + `map.yaml`: 2D occupancy map for Nav2's map_server
   - `objects.yaml`: named objects with position and size
   - `preview.png`: top-down render for a visual check
   The user picks one at launch: `env:=test/table_room`.
2. Sources, in order:
   - `test/*`: hand-built rooms from simple shapes (walls, table, blocks). First, to unblock Phase 3.
   - `robocasa/*`: RoboCasa kitchens (MuJoCo-native, labeled fixtures and objects; code MIT, assets CC BY 4.0).
   - Extensions, later: phone scan (`scan/*`) and generated-from-a-sentence rooms.
3. Floors stay flat: the all-joints walking policy was trained on flat ground only.

**Consequences:** `scan_pipeline/` becomes an optional future source. The phone-scan demo is no longer on the critical path.

**RoboCasa notes (2026-11-08)**
- Installed outside the repo in `~/hsim_third_party/` with its own Python env; only the exporter uses it. The sim, viewer and ROS never import RoboCasa.
- Assets are ~25 GB extracted (the "~10 GB" in their docs is the download size).
- RoboCasa pins `scipy==1.15.3`, whose Mac build doesn't load on macOS 27; its env uses scipy 1.16.3 instead. Export-time only, so low risk.
- The occupancy map is made by rasterizing every solid shape that overlaps the robot's height band (0.05–1.5 m). Ray-casting from above missed RoboCasa's 3 m walls.
- Spawn: RoboCasa's robot spot, turned to face the open room, moved out to 0.5 m clearance.
