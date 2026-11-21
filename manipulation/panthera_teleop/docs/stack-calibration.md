# Calibrate a bench for IK stacking

The stacking dashboard can estimate the overhead and wrist camera mounts from RGB images and the selected arm’s encoders. You supply the cameras’ lens specifications, the table height and the gripper setpoints. The estimate uses the stock Panthera arm geometry and the approximate camera locations in the two-arm bench model.

## Configure the bench

Copy the [automatic calibration template](../config/calibration/stack.example.json) to `config/calibration/stack.json`. It enables `vision.auto_calibrate` for the left arm. Use the `right` key for a right-arm setup or supply both arms separately. All distances and camera poses for an arm use that arm’s `base_link` frame, with Z pointing up.

| Field within each arm | What to enter |
| --- | --- |
| `wrist_camera` | The selected arm’s camera key from the [recording configuration](../config/recording.json). Cover the lens and check which feed changes to confirm the USB port assignment. |
| `gripper_open_rad`, `gripper_closed_rad` | Measured SDK encoder setpoints for open and empty-closed jaws, in radians. A block held between the pads must leave the encoder between them. |
| `vision.table_z_m` | Height of the level table surface relative to the selected arm base, in meters. |
| `vision.checkpoint` | Path to the trained body/stud adapter. Relative paths start in the configuration file’s directory. The [perception setup](../../panthera_sim/perception/README.md) covers the weights. |
| `vision.landmark_checkpoint` | Optional separate landmark head. The detector also looks for `landmarks.pt` beside the adapter. |
| `vision.overhead`, `vision.wrist` | Each camera’s native capture `resolution` as `[width, height]` and its vertical field of view in degrees as `fovy_deg`. Use the field of view at that resolution and crop. |

The template uses the modeled vertical fields of view, 80° overhead and 70° wrist. Check these against the lens specifications. If you already have a camera's OpenCV intrinsics, supply `camera_matrix` and `distortion` for the capture resolution in place of `fovy_deg`. A diagonal or horizontal field of view entered as `fovy_deg` changes the predicted image scale. Keep the camera focus and crop fixed during calibration and stacking.

## Check the camera layout

The estimator starts at the two-arm bench model’s mount locations. The overhead camera sits near the midpoint between the arm bases, about 37 cm above the selected base, looking along base +X and down at 45°. The wrist camera sits roughly 12 cm behind and 10 cm above the `grip_site` in the gripper frame, looking forward and down at 20°. In both nominal mounts the right side of the image points along the frame’s −Y axis. Matches refine those approximate mounts. The estimator rejects a proposed fit more than 12 cm or 25° from its nominal mount. For a substantially different layout, use [measured camera poses](#use-measured-camera-poses).

## Capture calibration views

1. Place two separated, differently colored upright 2×1 Mega Bloks on the bench. Keep the overhead camera, the arm base and both blocks stationary. Attach the wrist camera rigidly so it follows the gripper. Choose an arm pose where both cameras can see the blocks and the overhead camera can see the arm.
2. [Start the stacking dashboard](../README.md#stack-two-mega-bloks-with-ik), select the arm, then choose **Connect**. The calibration controls take images and read encoders. You reposition the arm between captures. Use positions that remain stationary with hands clear of the image and travel path.
3. Choose **Capture pose** after the arm settles. Change the gripper position by at least 8 mm or its orientation by at least 4°, then capture again. Keep recognizable arm features in the overhead frame and both stud rows visible from the wrist. Read the result for each camera after each capture. If a camera has too few matches, improve its view and capture a different pose.
4. Settle the arm at a survey pose where both cameras see both blocks and the gripper is at least 8 mm or 4° from every saved pose. Choose **Finish calibration** to check the fitted mounts against this fresh image pair. If the check fails, try another distinct survey pose or add more views with **Capture pose**. Once the fits pass, the dashboard freezes the survey images for block selection. Keep the arm and blocks in place until **Run stack** takes its own measurements.

The fitted mounts belong to this connection. If the overhead camera, arm base or wrist mount shifts, reconnect and collect new views before planning another stack.

## Check the physical setup

Verify that the measured block centers in **Preview plan** agree with independent locations on the bench. A 2×1 Mega Blok has 32 mm stud pitch in the model. On a resting block, the modeled center of a stud is 49.25 mm above the table. The dashboard rejects a block whose measured body-center height differs by more than 8 mm from its expected resting height. Comparing XY at known locations helps catch horizontal errors that this height check cannot detect.

The planner assumes stock Panthera joint order, signs and encoder zeroes. Its `grip_site` lies 165 mm along the positive X axis of Link6. Compare the modeled and measured tool positions at several reachable poses, and check that the physical jaw centers match the modeled pads. The planner uses a 40 mm block body, 18.5 mm studs and a 42.5 mm rise between seated layers.

## Use measured camera poses

Existing fixed-calibration configurations remain usable. To set one up, copy the [fixed calibration template](../config/calibration/stack-fixed.example.json). Leave `vision.auto_calibrate` absent or set it to `false`. Use this mode for a rig whose mounts fall outside the automatic estimator’s search region.

1. Calibrate each lens from checkerboard images covering the field of view at capture resolution. Enter its 3×3 OpenCV `camera_matrix` and its `distortion` vector in OpenCV coefficient order.
2. Set the arm at the survey pose. Measure known board points in the selected arm base frame, in meters. Fit each camera to these points with OpenCV’s `solvePnP`. Invert its object-to-camera transform to obtain `world_from_camera`. If the points use board-local coordinates, also compose the board-to-base transform.
3. Enter each `world_from_camera` as a 4×4 rigid transform. Its rotation maps optical right, down and forward axes into the selected arm base axes. Its translation is in meters.
4. Record the six wrist-arm joint angles in SDK order as `vision.wrist.reference_q_rad`. Return each joint to within 0.01 rad of that pose before the dashboard captures the survey.
