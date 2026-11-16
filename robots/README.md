# robots

One folder per physical robot we build: its model (URDF/MJCF from CAD), every number with its source, checks, and
hardware findings.

| Robot | Status |
|---|---|
| [`wisco_v1/`](wisco_v1/README.md) | the WiscoHumanoids humanoid: modelled from CAD + MyActuator datasheets, walking policy trained in sim, hardware bring-up starting |

The Unitree G1, used as a reference robot in simulation, has no folder here: its model comes from mjlab
(`assets/robots/g1/`).
