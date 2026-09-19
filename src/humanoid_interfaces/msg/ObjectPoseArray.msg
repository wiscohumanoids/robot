# All objects perception currently believes are in the scene.
#
# Published by: perception (stub today: one hardcoded cube; ORB-SLAM3 +
#               FoundationPose wiring later), on /object_poses at ~30 Hz
# Consumed by:  task_planner, behavior_tree (and manipulation, once it needs it)
#
# header.frame_id is "map" by contract: perception transforms into the map frame
# itself (via TF) so consumers do not have to.

std_msgs/Header header
ObjectPose[] objects
