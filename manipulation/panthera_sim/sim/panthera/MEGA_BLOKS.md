# First Builders style scene

`sim.mega_bloks.build_mega_bloks_scene(output_path, seed=0, count=8)` writes a MuJoCo scene with two Panthera arms on the wood bench. Use `count=0` for an empty bench, or up to 12 blocks. Seeds reproduce separated tabletop placements, colors and yaw. `randomize=False` uses a fixed grid.

The scene retains the approximate hardware camera mounts and wood appearance from the `photo_20261027` setup in RobotArmLearning. The cameras are named `overhead`, `left_wrist` and `right_wrist`. These estimated poses do not provide real-camera calibration.

Each block has a free joint and a rectangular body geom named `mega_block_NNN_body`. Tapered stud geoms are named `mega_block_NNN_stud_K`. The returned metadata gives the corresponding names, color, stud-grid shape and dimensions in metres. This separates visible body surfaces from studs for segmentation labels.

| Parameter | Approximation |
| --- | --- |
| Stud pitch | 32 mm |
| Body height | 40 mm |
| Stud projection | 18.5 mm |
| Stud diameter | 22 mm |
| Body footprints | 31.6 × 31.6, 63.6 × 31.6, 95.6 × 31.6, 63.6 × 63.6 mm |

[Mattel’s First Builders play guide](https://mattel.co.jp/fisher_price/megabloks/firstbuilders/images/play-guide.pdf) shows a 3.2 × 3.2 cm single-stud block with a 5.9 cm overall height. [Argos’s supplier Q&A](https://www.argos.co.uk/product/9082603) gives approximately 31.6 mm width and 58.5 mm height. The model follows that footprint and total height. The body-to-stud height split, stud pitch and larger footprints are estimates. The projecting studs have a mild taper and top bevel. The solid body and studs support rendering and rigid contact. Snap-fit assembly would require underside sockets and a coupling model.

The bench top is at `z=0.05` m. Spawn origins range from `x=0.17` to `0.48` m and `y=-0.23` to `0.23` m. The body origin is its rectangular body's center, with the studs extending along its local positive z axis. Block colors use plastic materials with specular highlights. Meshes and the wood texture come from the adjacent checked-in assets; generated XML uses absolute paths so it can be loaded from a dataset output directory.

From `manipulation/panthera_sim`, open the bench in the MuJoCo viewer. The Control panel exposes the joints and grippers on both arms.

```sh
.venv-dino/bin/python -m sim.mega_bloks --view --count 8
```

For a headless render of the overhead camera, run

```sh
MUJOCO_GL=egl .venv-dino/bin/python -m sim.mega_bloks --count 8 --seed 0 --preview /tmp/mega_bloks.png
```

The command prints the XML and preview paths. Initialize physics or render state with `mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)` to use both arms’ home pose and preserve the blocks’ initial free-joint poses.
