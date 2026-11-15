# hsim_localization

Localization now lives in `hsim_navigation` (node `localization`, owns `map → odom`): AMCL on a known map
(`mapping:=map`), slam_toolbox (`mapping:=slam`), or RTAB-Map (`mapping:=rtabmap`). Without the autonomy stack,
`sim_stack.launch.py` fixes `map → odom` at the env's spawn pose.
