#!/bin/bash
# gnome-terminal --tab -- roslaunch realsense2_camera rs_l515.launch align_depth:=true
# sleep 3
# gnome-terminal --tab -- roslaunch apriltag_ros continuous_detection.launch camera_name:=/camera/color/ image_topic:=image_raw
# sleep 3
# gnome-terminal --tab -- roslaunch interbotix_sdk arm_run.launch robot_name:=rx200 use_time_based_profile:=true gripper_operating_mode:=pwm
# sleep 5

gnome-terminal --tab -- roslaunch armlab.launch launch_station:=true
sleep 2
./control_station.py -p config/rx200_pox.csv -e config/latest_calibration.txt 