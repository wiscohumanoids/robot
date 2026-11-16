# Contributing

This repo is shared by every WiscoHumanoids software team. The rules below keep it one stack that fits together.

## Ground rules

1. **Interfaces are shared; internals are yours.** The topics, actions, services, TF edges, message fields and skill
   names in `ros2_ws/src/hsim_bringup/config/interface_contract.yaml` (tables: `docs/interface_contract.md`) belong
   to everyone. Inside your own package, do what you like.
2. **One publisher per topic and one owner per TF edge.** If you need to publish something the contract gives to
   another node, you are proposing a contract change.
3. **Stub first, real second.** A stub and the real node it becomes share a ROS node name, so the contract doesn't
   change when one replaces the other. Replace a stub only with something that honors the same contract entry.
4. **Say what's fake.** Stubs say so in their docstring and README; privileged data (`/sim/ground_truth/*`) is only
   for graders in `tools/eval/` and clearly labeled stubs (decision 0007).
5. **One source for every number.** Robot numbers live in `robots/wisco_v1/model/robot_numbers.yaml` with their
   source; generated files (`robot.urdf`, `robot.xml`, `model_report.json`, the training MJCF,
   `docs/interface_contract.md` tables) are never hand-edited: change the source and regenerate.
6. **Record decisions.** A change of direction gets a short decision record in `docs/decisions/`; experiment logs go
   in the relevant work log (`docs/runs/`, `training/locomotion/wisco_v1/WORK_LOG.md`); hardware findings for the
   mechanical/electrical team go in `robots/wisco_v1/HARDWARE_ISSUES.md`.

## Everyday workflow

```bash
tools/ros.sh up && tools/ros.sh shell                         # ROS 2 Jazzy container (infra/docker), repo mounted at /hsim
colcon build --symlink-install && source install/setup.bash   # in /hsim/ros2_ws
ros2 launch hsim_bringup autonomy.launch.py env:=test/table_room perception:=stub mapping:=map   # whole stack, repeatable
ros2 run hsim_bringup contract_check --phase 3                # one publisher? right type? fast enough?
```
To develop one layer against the rest, launch the stack without that node (each launch file's arguments switch
layers on and off, e.g. `planner:=false`), run yours under the same node name, and run `contract_check`.

## Changing an interface (topic, type, rate, owner, TF edge, action, service, message field, skill)

1. Edit `interface_contract.yaml` (and the `.msg`/`.srv`/`.action` file, registered in `hsim_interfaces/CMakeLists.txt`).
   Phase 1 entries are frozen: changing one needs a decision record.
2. `python3 tools/gen_docs.py` to refresh the generated tables.
3. Update the stub **and** every consumer in the same change.
4. Run the checks below.

Prefer adding fields at the end of a message, or a new message, over changing or removing fields: field changes break
recorded bags and every node built against the old definition.

## Adding a node or package

1. Create the package under `ros2_ws/src/` (`ament_python` for Python, `ament_cmake` for C++; copy an existing
   package for the boilerplate). Keep decision logic in a ROS-free module and unit-test it in `<pkg>/test/` with
   plain pytest (no `rclpy` imports in tests).
2. Add its topics, TF edges, actions and services to `interface_contract.yaml` with its node name as the owner.
   Publish topic names as string literals (`create_publisher(Msg, '/topic', ...)`): `tests/test_contract.py` reads them.
3. Start it from the right launch file in `hsim_bringup/launch/`.
4. Write the package README: what it does, node names, interfaces, status (real / stub / planned), how it was tested.

## Checks

| Check | Command | When |
|---|---|---|
| Static: contract vs code, interface files, docs and links, robot profiles and manifests, unit tests | `python -m pytest tests ros2_ws/src/*/test` | every change (CI: `static`) |
| Generated docs are current | `python3 tools/gen_docs.py --check` | every change (CI: `static`) |
| Build + stack smoke tests (contract check + drive test) | `tools/ros.sh test stub`, `tools/ros.sh test sim test/table_room [robot]` | every change touching ROS code (CI: `ros`) |
| Autonomy checks (navigation, behavior tree, reach) | `tools/ros.sh autotest [name] [robot]` | changes to navigation, perception, behavior, planner, locomotion |
| Robot model regenerates cleanly | `robots/wisco_v1/README.md`, "Rebuild" | changes to `robots/wisco_v1/model/` |

## Common pitfalls

- Generated files: see rule 5. `robots/wisco_v1/original_from_team/` is the CAD export as received; never edit it.
- `ros2 launch` gives nodes no TTY, so keyboard input only works from `ros2 run` or the Mac viewer.
- Wisco V1 has no lidar: its stereo camera renders the built world, so sim runs with Wisco need `env:=`.
- ros2_control controllers and hardware plugins run inside the real-time loop: no allocation, locking or logging in
  `update()` / `read()` / `write()`.
- Commit messages, docs and comments describe the work, not who did it; authorship lives in git.
