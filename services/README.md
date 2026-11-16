# services (GPU VM)

Network inference endpoints the ROS nodes call, ROS-agnostic on purpose. Reached from the Mac through SSH tunnels.

| Service | Called by | What |
|---|---|---|
| [`perception_server/`](perception_server/README.md) | `perception` node (`perception:=server`), port 8100 | open-vocabulary detection: Grounding DINO + SAM 2.1 + depth → 3D boxes |
| [`vla_server/`](vla_server/README.md) | `manipulation` node (`/skills/manipulate`) | GR00T policy server: image + joint state + instruction → action chunk |
