# 0003 — Perception runs on the GPU VM as a service

**Context:** The brief puts everything except training and the VLA on the CPU VM. Grounding-DINO, SAM2 and FoundationPose are GPU models; on CPU they take seconds per frame.

**Update 2026-11-10:** built as Grounding DINO (base + tiny ensemble) + SAM 2.1 + ZED X depth. FoundationPose dropped: the VLA
grasps from camera images, so 6-DOF object pose isn't needed. SAM 3 not used (gated license).

**Decision:** Run them in `services/perception_server` on the GPU VM, next to the VLA server. `hsim_perception` on the CPU VM is a thin ROS client. The server has a fake mode that returns ground-truth poses from the sim, so Phases 3–4 can be built without the GPU VM running.
