# One detected object. Element of ObjectPoseArray.objects.
# The pose is expressed in the frame named by the enclosing ObjectPoseArray.header.

string object_id            # stable id across frames, e.g. "cube_1"
string label                # semantic class, e.g. "cube"
geometry_msgs/Pose pose
float32 confidence          # 0.0-1.0
