# Wisco V1 model sources

Where every number in the Wisco V1 model comes from, and the scripts that write them into the generated files.
The whole flow, with the table of values and their sources, is in [`../README.md`](../README.md).

| File | What |
|---|---|
| `robot_numbers.yaml` | **the numbers file**: motors (datasheet values with sources), delay, bus, mass overrides, gains and control modes, default pose, locomotion joints, sensor poses |
| `motor_datasheets/` | MyActuator parameter sheets the motor values come from |
| `robot_params.py` | derives every per-joint value (gains, limits, friction, damping, armature, torque-speed) from the numbers file |
| `sync_robot_folder.py` | writes those values into `../robot.urdf`, `../robot.xml`, `../model_report.json` (idempotent) |
| `build_training_model.py` | `../robot.xml` → `wisco_v1_train.xml` + `train_model_info.json`, the model mjlab trains on |
| `linkage_tools.py` | ankle linkage reach and Jacobian analysis → `ankle_linkage.json` |
| `ankle_motor.py` | ankle joint torque → per-motor torque |

Edit only `robot_numbers.yaml` (and the CAD-derived inputs); everything else here is generated or derived.
