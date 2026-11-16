# Panthera + Quest passthrough

Both arms and the Quest 3S were connected and tested. The external HD USB camera also passed a capture test; passthrough uses the headset's own cameras.

## Controls

- Left controller operates the arm on hub port 2 (USB 1-1.2); right controller operates the arm on hub port 3 (USB 1-1.3). These assignments were swapped at the operator’s request.
- Hold the side grip and move/rotate the controller to move that arm.
- Both arms use the original manufacturer rotation and translation mapping.
- Release the side grip to hold the arm's target position.
- Hold the side grip and squeeze the trigger to close the gripper; ease off the trigger to open it.
- X on the left controller or A on the right resets that arm to its starting pose.
- The official demo moves both arms at startup and returns them to zero on normal exit (Ctrl+C). Ctrl+C is not an emergency stop.

## Connections

- Keep robot USB cables in the verified front communication sockets. The other front socket exposes a UART; the upper socket exposes a hub.
- Keep robot USB cables in their existing external hub ports: port 2 is assigned to the left controller and port 3 to the right controller. The saved mapping uses these connections because both boards report the same serial number.
- Robot power supplies must be powered, and the round motor-power buttons must be on. Fans alone did not indicate motor power.
- Quest connects to the laptop by USB data cable. Accept USB debugging if prompted.

## Start another session

1. Power and connect both arms as above, with the workspace clear and grippers empty.
2. Run `start-headset.sh` beside this file. In the connection UI choose USB and Start Server. Use UDP IP `127.0.0.1`, port `5005`. Do not start a second bridge if one is already running.
3. In the Quest browser open `http://localhost:8080/index.html` and select **Start Passthrough**. Pick up both controllers with side grips released.
4. Run `start-teleop.sh` for configuration checks only. Run `start-teleop.sh --run` to start live control, including the automatic starting movement.

Launch scripts are now in `scripts/`. The launch scripts use the isolated software installed under this task's `work` folder. Keep that folder alongside `app`, `config`, and `scripts`. If the headset USB disconnects, its USB forwarding needs to be restored by restarting the connection bridge before restarting robot control. If robot USB hub ports change, the mapping needs to be identified again.

This setup uses the manufacturer's control loop, including its startup/reset/exit movements. No independent emergency-stop or loss-of-tracking safety system was added or validated.

## Physical handle mode

Run `../start.sh` from `docs/` (or `./start.sh` from the workspace root) to guide both arms independently by their attached handles, with gravity assistance. Each arm moves only from physical guidance; neither follows the other. Quest controllers are unused. There is no automatic starting pose or return-to-zero trajectory in this mode. Grippers are passive. Support the handles: gravity assistance does not lock an arm at a fixed pose. Support the arms before Ctrl+C, which ends assistance and sends the SDK stop command (the SDK also applies its configured exit behavior). Never run hand guidance and Quest teleop simultaneously. Running `start-hand-guide.sh` without `--run` only checks configurations.

Hand guidance stops both arms if either reports an out-of-range joint or motor fault. Keep supporting the handles and avoid extreme wrist/joint positions. A fault exit logs the exact reason in `logs/hand-guide-events.log`; do not widen limits to work around it.
