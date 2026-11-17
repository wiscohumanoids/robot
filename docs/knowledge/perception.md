# Perception, End to End — The Whole Rant

Perception is genuinely the layer we've explored least, so let me build it up properly. This is a big topic so I'll go in order: what SLAM actually is and how ORB-SLAM3 works internally, then loop closure, then how objects get placed in a room, then the ROS integration and TF question, then the practical "how much does a perception team build vs use" answer, and finally the stubbing/parallel-development question.

## What SLAM actually is

SLAM stands for Simultaneous Localization and Mapping, and the name contains the whole chicken-and-egg problem it solves. To know where you are (localization), you need a map. To build a map, you need to know where you are. SLAM does both at once, bootstrapping each from the other.

Concretely: a camera moves through a room. From the stream of images alone (plus maybe an IMU), SLAM figures out two things simultaneously — the camera's trajectory through 3D space (where it was at every moment) and a 3D map of the room (a cloud of points marking where physical features are). Neither is known ahead of time; both fall out of the same optimization.

The core insight that makes it possible: if you see the same physical corner of a table from two different camera positions, the way that corner shifts between the two images constrains both where the camera moved AND where the corner is in 3D. Stack up thousands of these constraints across hundreds of features and camera positions, and you can solve for everything at once. That's SLAM.

## How ORB-SLAM3 works internally

ORB-SLAM3 is the current standard feature-based visual SLAM system. "Feature-based" means it doesn't use whole images — it tracks a sparse set of distinctive points. Let me walk the pipeline.

ORB features — the "what it tracks." ORB is a type of feature detector. In each image it finds a few hundred distinctive keypoints — corners, blobs, high-contrast spots that are easy to recognize again from a different angle. For each keypoint it computes a descriptor, a compact binary fingerprint of the local image patch around that point. The magic property: the same physical corner, seen from a slightly different position or rotation, produces a similar descriptor. So descriptors let you match "this corner in frame 50" to "the same corner in frame 51." ORB is chosen because it's extremely fast to compute and match, which is what lets SLAM run in real time.

The system runs as three parallel threads. This is the architecture to hold onto:

Thread 1 — Tracking (fast, every frame, ~30Hz). For each new camera frame: extract ORB features, match them against the features it's currently tracking from the local map, and from those matches compute where the camera must be right now. This is the "localization" half, running at camera framerate. It's fast because it only looks at a small local set of nearby map points, not the whole map. If tracking is good, it also decides whether this frame is interesting enough to become a keyframe — a saved reference frame that gets added to the map.

Thread 2 — Local Mapping (slower, per keyframe). When a new keyframe is added, this thread does the heavier work: triangulate new 3D map points from the keyframe's features (using the shift between this keyframe and nearby ones to compute depth), and run local bundle adjustment — an optimization that jointly refines the positions of the recent keyframes and the map points they see, minimizing the reprojection error (the gap between where a 3D point should appear in an image and where it was actually observed). This is where the map gets refined and made consistent locally. It runs behind tracking, so it doesn't block the fast loop.

Thread 3 — Loop Closing (slowest, event-driven). This watches for the robot revisiting a place it's been before, and corrects accumulated drift when it does. This is important enough to deserve its own section.

## Loop closure — the thing that makes SLAM not drift forever

Here's the fundamental problem SLAM has: drift. Every frame, tracking has tiny errors. Frame to frame those errors are minuscule, but they accumulate. After walking around a building for five minutes, the estimated trajectory might be off by meters — the map slowly bends and warps because thousands of tiny errors compounded. Pure frame-to-frame tracking (called visual odometry) drifts without bound.

Loop closure fixes this. The idea: if the robot returns to a place it has been before, and recognizes it, that recognition is a powerful constraint. "I am now at the same spot I was at ten minutes ago" — if the map says those two spots are two meters apart, you know you've accumulated two meters of drift, and you can snap it back.

How does it recognize a revisited place? This is the clever part, called bag of words. ORB-SLAM3 pre-builds a vocabulary of visual "words" — clusters of common descriptor patterns, learned offline from lots of images. Every image can then be summarized as a histogram of which visual words it contains, like a document summarized by its word frequencies. To detect a loop, the system asks: "does the current frame's bag of words closely match the bag of words of any old keyframe?" This is a fast database lookup, not a pixel comparison. If a strong match comes back, it's a loop candidate — the robot has probably been here before.

Then it verifies and corrects. A bag-of-words match could be a coincidence (two different hallways that look alike), so it geometrically verifies — do the actual features line up consistently? If yes, it computes the drift (the discrepancy between where the map thinks the robot is and where the loop says it is) and runs pose graph optimization: it treats the whole trajectory as a graph of keyframes connected by constraints, adds the new loop-closure constraint, and redistributes the accumulated error smoothly across the whole loop. The map visibly "snaps" into consistency — a warped hallway straightens out. After a big loop closure it may run a full global bundle adjustment to refine everything.

The "Atlas" and multi-map part (what the "3" in ORB-SLAM3 adds). ORB-SLAM3's headline feature is robustness when tracking is lost — camera covered, sudden motion, blank wall. When it loses tracking, instead of failing, it starts a new map. Later, if it recognizes a place connecting the new map to an old one, it merges them. This collection of maps is the "Atlas." It makes the system far more robust to the real-world messiness of a camera getting confused.

## Placing objects in the room — perception beyond SLAM

Now, SLAM gives you the robot's position and a sparse point-cloud map of geometric features. But those points are just anonymous 3D dots — SLAM doesn't know "that cluster of points is a cube." Putting semantic objects in the room is a separate job layered on top, and this is where your manipulation and high-level needs come in.

The pipeline: an object detector or pose estimator (like FoundationPose, which we covered) runs on the camera image and says "there's a cube, and its 6-DOF pose relative to the camera is this." SLAM tells you where the camera is in the map. Compose those two transforms and you get "the cube is at this position in the map frame." Now the object is placed in the room's coordinate system, and it stays there even as the robot moves, because it's anchored to the map, not the camera.

This is how the behavior tree's blackboard gets "cube at (0.5, 0.2, 0.8) in the map" — perception detects it in the image, SLAM provides the camera-to-map transform, and the composition lands the object in world coordinates. Build this up across many objects and you get a semantic map or scene graph — the room represented not as anonymous points but as labeled objects with poses, which is exactly what a task planner needs to reason about ("navigate to the table, the cube is on it").

## The ROS integration and the TF question

You asked whether SLAM runs super fast and always updates TF. Here's the precise picture, and it connects to the multi-writer TF question from before.

SLAM does not run uniformly fast — remember the three threads run at different rates. Tracking runs at camera framerate (~30Hz), local mapping runs slower per keyframe, loop closing fires only occasionally. What SLAM publishes to the rest of the system is the robot's pose in the map, and it publishes a specific edge of the TF tree.

Recall the TF chain from before:

```
map → odom          ← SLAM owns this edge
odom → base_link    ← state estimation / odometry owns this edge
base_link → camera  ← static, from mounting
```

Here's the elegant division of labor, and it directly answers your question. State estimation (a fast filter over IMU and odometry: an IEKF, or in our stack leg odometry + an EKF; our edge is odom → base_footprint, the floor point under the pelvis) publishes odom → base_link at high rate — this is smooth, fast, locally accurate, but drifts over time. SLAM publishes map → odom — this is the drift correction, updated more slowly and jumping when loop closures happen. The robot's position in the map is the composition of both: the fast smooth local motion plus the slow global correction.

Why split it this way? Because you want both properties. The fast odom → base_link gives you smooth, high-rate motion that never jumps (critical for control — you don't want your balance controller seeing the robot teleport). The slow map → odom gives you global consistency and corrects drift (critical for navigation — you want to actually reach the table). When a loop closure fires and SLAM realizes there's been two meters of drift, it corrects map → odom — the global position snaps, but the smooth local motion in odom frame is undisturbed. Control keeps running smoothly off odom; navigation benefits from the corrected map. This is a genuinely beautiful piece of design and it's the standard ROS convention (REP 105 if you want the spec).

So: SLAM doesn't update TF at a blazing uniform rate — it updates its one edge (map → odom) when it has a correction, while state estimation updates its edge (odom → base_link) fast and smooth. Two writers, two different edges, no conflict, exactly the ownership principle from before.

## How much does a perception team build vs use

This is your practical lead question and the answer is encouraging: perception is one of the most library-heavy layers — you use enormous amounts of existing implementation and build relatively little.

Black-box / use directly:

ORB-SLAM3 itself — this is a mature research codebase you run, not write. You do not implement SLAM. You compile it, configure it for your camera, and consume its output. There are ROS2 wrappers that publish its pose and map to topics.
FoundationPose — pretrained, you run it.
The camera driver — realsense-ros or similar, publishes images to topics. Use as-is.
TF / tf2 — pure library.
Object detectors (YOLO, SAM, Grounding-DINO) — pretrained models you run.

Build yourself (the real work):

Camera calibration — measuring your camera's intrinsics (lens parameters) and the extrinsics (where the camera is mounted relative to the robot base). This is careful one-time work and if you get it wrong everything downstream is wrong. Genuinely important.
The ROS integration glue — wiring ORB-SLAM3's output and FoundationPose's output into your TF tree and your custom /object_poses topic, in your frames, at the right rates. This is where perception-team effort actually goes.
The semantic layer — composing SLAM pose + object detection into "objects placed in the map," building the scene graph the planner consumes. This is your robot-specific contribution.
Tuning — ORB-SLAM3 has many parameters (number of features, keyframe thresholds) that need tuning for your camera and environment.

So the shape mirrors the rest of the stack: the hard algorithmic core (SLAM's optimization, the loop closure math, the object detection networks) is library-provided and you'd be crazy to reimplement it. Your team's job is calibration, ROS integration, the semantic composition, and tuning. They need to understand the internals — enough to debug why tracking is lost or why loop closure isn't firing — but they don't implement them. That's the right level: understand deeply, implement selectively.

How deep should they understand? Enough to debug. When ORB-SLAM3 loses tracking constantly, they need to know it's probably a feature-poor environment (blank walls) or motion blur (moving too fast) — that requires understanding the tracking thread. When the map drifts and never corrects, they need to know loop closure isn't triggering — that requires understanding bag-of-words recognition. So the internals I explained above are exactly the debugging vocabulary they need. Not to build SLAM, but to make library SLAM actually work on your robot.

## Testing with a real camera, and the stubbing/parallel question

Yes, this is very doable and it fits the scaffold cleanly. Here's how it slots in.

Testing with a real camera in isolation. A perception team does not need the rest of the robot to make progress. They need a camera (a RealSense or similar) and a laptop. They can run the camera driver, run ORB-SLAM3, walk the camera around a room, and watch it build a map and localize in RViz (ROS's visualization tool) — all standalone, no robot, no other teams. This is actually one of the most independently-developable parts of the stack because perception's input is a camera feed, which you can get from a handheld camera or even a recorded bag file. They can literally record a camera walking through your lab once, and then develop against that recording forever without even needing the camera live. RViz will show the point cloud map growing, the trajectory, the detected objects — the "is the SLAM-ing actually working" visualization you asked about is exactly what RViz gives you for free.

How it was stubbed in the scaffold. Recall from the scaffold discussion: perception's stub publishes /object_poses with a hardcoded object position and publishes a fake map → odom transform (identity, no drift). That stub let the task planner and behavior tree develop against "there's a cube at a known spot" without real perception existing. Meanwhile the real perception team develops their real node against a camera or a bag file, in parallel, in total isolation. When their node is ready, it replaces the stub — it now publishes real /object_poses and a real map → odom — and because it honors the same interface (same topic, same message type, same TF edge), nothing downstream changes. The behavior tree that was reading hardcoded object poses now reads real ones and doesn't know the difference.

Does it fit the parallel-development structure? Perfectly — arguably better than most nodes, because perception is so input-isolated (it just needs a camera feed) and its outputs are cleanly defined (a TF edge and an object-poses topic). The perception team can go from zero to "SLAM working on a real camera, visualized in RViz, objects detected and placed" without ever touching the rest of the robot, then integrate by swapping their node in for the stub. It's one of the cleanest parallel workstreams you have.

## The whole perception picture in one breath

SLAM solves the chicken-and-egg of needing a map to localize and needing to localize to map, by tracking sparse ORB feature points across frames and jointly solving for camera trajectory and 3D map. ORB-SLAM3 runs three threads — fast tracking at framerate, slower local mapping with bundle adjustment per keyframe, and event-driven loop closing. Drift accumulates frame to frame without bound, and loop closure fixes it by recognizing revisited places via bag-of-words matching and redistributing the error through pose graph optimization, snapping the map back into consistency. Objects get placed in the room by composing SLAM's camera-in-map pose with a detector's object-in-camera pose, building a semantic map the planner can reason about. In ROS, SLAM owns the map → odom TF edge (slow drift correction) while state estimation owns odom → base_link (fast smooth local motion), giving you both smoothness for control and global consistency for navigation without the two writers conflicting. A perception team uses ORB-SLAM3, FoundationPose, and detectors as libraries — building calibration, ROS integration, the semantic layer, and tuning rather than the algorithms themselves — and can develop the whole thing in isolation against a real camera or a recorded bag, visualizing it live in RViz, then drop their node into the scaffold in place of the stub with zero disruption downstream. It's one of the most independently-developable and library-heavy layers in the entire stack.

## In our stack

| Concept above | What Wisco V1 uses (this repo) |
|---|---|
| Camera | ZED X stereo RGB-D on the head (no lidar on Wisco; the G1 reference robot has one). In sim: `stereo_camera` with a stereo depth-error model |
| SLAM (`map → odom`) | RTAB-Map visual SLAM by default (`mapping:=rtabmap`); slam_toolbox (`mapping:=slam`) and AMCL on a known map (`mapping:=map`, tests) are the alternatives. Node `localization`, `ros2_ws/src/hsim_navigation/` |
| State estimation (`odom → base_footprint`) | leg odometry (stance-foot kinematics) + `robot_localization` EKF at 50 Hz, `ros2_ws/src/hsim_state_estimation/` |
| Object detection | open-vocabulary: Grounding DINO + SAM 2.1 on the GPU server, depth back-projected to 3D boxes (`services/perception_server/`, client node `perception` → `/perception/detections`, camera frame). FoundationPose-style 6-DOF poses are not used yet |
| Placing objects in the room | `world_model`: detections → map frame with the robot's own pose, tracked and merged over views → `/world/objects`, `/world/summary` (what the planner reads) |
| Stubbing for parallel work | `perception:=stub`: line-of-sight detections from the simulator (privileged positions, honest visibility), same topic and node name |

Measured in sim: 3D error 1–4 cm median, 0.4–0.8 s per frame; wrong labels are the weak spot. Topics and frames: `docs/interface_contract.md`.

