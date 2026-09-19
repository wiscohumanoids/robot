# Contributing

## Ground rules

1. **Interfaces are frozen; internals are yours.** The topics, actions, TF edges,
   message fields and skill names in [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md)
   are shared by every team. Inside your own package, do what you like.
2. **One publisher per topic and per TF edge.** If you need to publish something
   the contract gives to another node, you are proposing a contract change.
3. **Stub first, real second.** Replace a stub only with something that honors
   the same contract entry, then check it (below).
4. **Say what's fake.** A new stub says `STUB` in its docstring and README, and is
   listed in the contract with `status: stub`. Don't let a placeholder read as real.

## Who owns what

Areas are listed in [`STATUS.md`](STATUS.md#who-owns-what) (planning, perception,
state-estimation, navigation, locomotion, manipulation, low-level, safety,
integration). **TODO for the integration lead:** record a named owner for each
area here, plus the interface-contract reviewer.

| Area | Owner | Reviewer for contract changes |
|---|---|---|
| planning | _unassigned_ | _unassigned_ |
| perception | _unassigned_ | |
| state-estimation | _unassigned_ | |
| navigation | _unassigned_ | |
| locomotion | _unassigned_ | |
| manipulation | _unassigned_ | |
| low-level | _unassigned_ | |
| safety | _unassigned_ | |
| integration (contract, CI, bringup) | _unassigned_ | |

## Everyday workflow: replacing a stub with your real node

```bash
./docker/run.sh                                                  # or your native ROS 2 Humble shell
colcon build --symlink-install && source install/setup.bash
ros2 launch bringup full_stack.launch.py perception:=external    # every other layer stays a stub
ros2 run my_perception_pkg my_perception_node                    # yours
ros2 run bringup check_contract.py                               # one publisher? right type? fast enough?
ros2 run bringup smoke_task.py                                   # whole task layer still works end to end
```
Add `sim:=false` to skip MuJoCo when your layer doesn't need the robot body.
Switch names for every layer are in [`src/bringup/README.md`](src/bringup/README.md).
Keyboard teleop needs its own terminal: `ros2 run teleop_input teleop_node`.

## Changing an interface (topic, type, rate, owner, TF edge, message field, skill)

1. Edit [`src/humanoid_interfaces/config/interface_contract.yaml`](src/humanoid_interfaces/config/interface_contract.yaml)
   (and the `.msg`/`.action` file, registered in that package's `CMakeLists.txt`).
2. `python3 tools/gen_docs.py` to refresh the generated tables (never hand-edit them).
3. Update the affected stub **and** every consumer in the **same PR**.
4. `python3 -m pytest tests src/*/test` must pass.
5. Get the PR reviewed by the interface-contract reviewer and tell every team that
   is listed as a consumer of what changed.

Prefer adding fields at the end of a message, or adding a new message, over
changing or removing existing ones: field changes break recorded rosbags and
every node built against the old definition.

## Adding a new node

1. Create the package (`ament_python` for Python, copy any existing stub package
   for the boilerplate). Keep decision logic in a ROS-free module and unit-test it
   in `<pkg>/test/` with plain `pytest`.
2. Add it to `interface_contract.yaml`: a `nodes:` entry (id, layer, `status`,
   package, executable, `launch_arg`, area, `replaces`, `summary`), plus its
   `topics:` / `actions:` / `tf:` entries with **yourself as the single owner**.
   Publish topic names as string literals (`create_publisher(Msg, '/topic', ...)`)
   and TF edges as `tf.header.frame_id = '...'` / `tf.child_frame_id = '...'`
   literals: the static tests read them.
3. Add its launch switch in `src/bringup/launch/full_stack.launch.py`.
4. Add the package to `bringup/package.xml` `exec_depend`s.
5. Write the package README (layer, status, rate, interfaces, what's fake, what
   replaces it) and `python3 tools/gen_docs.py`.

## Checks that must pass

| Check | Command | When |
|---|---|---|
| Static contract + consistency + unit tests | `python3 -m pytest tests src/*/test` | every PR (CI: `static`) |
| Docs match the contract | `python3 tools/gen_docs.py --check` | every PR (CI: `static`) |
| Builds + tests + stub stack honors the contract + end-to-end task | CI job `ros` (Docker) | every PR |
| Live contract check on your running stack | `ros2 run bringup check_contract.py` | before asking for review |

## Common pitfalls

- The 23-joint order and `NOMINAL_POSE` are duplicated across several files;
  `tests/test_repo_consistency.py` keeps them consistent, so change them together.
- xacro arguments are read with `$(arg name)`; `${name}` looks up a *property* and
  fails at launch time. The tests expand the URDF to catch this.
- `ros2 launch` gives nodes no TTY, so keyboard teleop only works from `ros2 run`.
- `lowlevel_control` runs inside `controller_manager`'s real-time loop: no
  allocation, locking or logging in `update()`.

## Not decided yet

No `LICENSE` file exists (package manifests say Apache-2.0), and there is no
`CODEOWNERS` (it needs real GitHub handles). The integration lead should decide both.
