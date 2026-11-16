# WisCortex academy

Interactive study notes for the whole stack. No dependencies and no server: open `index.html` in a browser.

Built from the knowledge base in [`../knowledge/`](../knowledge/): [end_to_end.md](../knowledge/end_to_end.md),
[perception.md](../knowledge/perception.md), [locomotion_rl.md](../knowledge/locomotion_rl.md) and
[manipulation.md](../knowledge/manipulation.md); the controls chapter (`src/ch_ctl.js`) is the source of
[controls.md](../knowledge/controls.md). The academy teaches the general concepts; for what Wisco V1 runs, read the
"In our stack" section at the end of each knowledge-base document.

## Chapters, in stack order
1. The map: how the layers fit together (`ch_map.js`)
2. Perception: SLAM, loop closure, objects and TF (`ch_see.js`)
3. Planning and navigation: task planner, behavior tree, Nav2 (`ch_body.js`, ids `b-planner`, `b-bt`, `b-cmdvel`)
4. Learning to walk: RL and PPO (`ch_walk1.js`, `ch_walk2.js`)
5. Arms and hands: IK, ACT, diffusion policy, VLAs (`ch_arms.js`)
6. Controls: PD for the legs, impedance control for the arm, Pinocchio and CRISP, where a WBC fits (`ch_ctl.js`)
7. Hardware interface: ros2_control, EtherCAT, FOC and CAN, the full chain (`ch_body.js`, ids `b-ros2c`, `b-ethercat`, `b-foc`, `b-capstone`)

The chapter order lives in `CHAPTERS` in `src/core.js`. Previous/next buttons follow that order, not file order.

## Other source files
- `src/core.js` DOM helpers, progress store, content blocks (`Widget`, `Slider`, `Seg`, `Check`, `Order`, `Match`)
- `src/pages.js` top bar, home page and its animated graphics, lesson page, spaced-review deck, router, and the `LAYERS` list behind the "one second of the stack" chart
- `src/style.css` black and red theme, light and dark. Each chapter has a color token (`--see`, `--plan`, `--walk`, `--arms`, `--ctl`, `--hw`)

## Build
```
node build.js
```
This writes `index.html` (standalone, kept in the repo so the page opens without building) and `artifact.html`
(a fragment for publishing, gitignored). Commit `index.html` together with any change to `src/`.

## Adding a lesson
Call `lesson({ id, ch, mins, title, lede, build(){ return [ ...blocks ] }, cards: [{ q, a }] })` in a chapter file,
add a short name for it to `SHORT` in `pages.js`, and rebuild. `Check`, `Order` and `Match` blocks are collected into
the collapsed "Practice questions" section at the end of the lesson.
If it belongs in the stack chart, add a row to `LAYERS` in `pages.js`.
