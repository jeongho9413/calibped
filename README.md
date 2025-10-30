# calibped

# Introduction

This `calibped` package v.1.1 performs pedestrian segmentation on LiDAR point clouds using LiDAR–camera calibration.
It has been tested with a **Livox Mid-360** and **GoPro Hero 13 Black** on **ROS 2 Humble**.

> ⚠️ Work in progress: The codebase is under active development and camera intrinsic parameters are still being tuned. Expect frequent updates.
> 

# Quick start

We assume that the command `source /opt/ros/humble/setup.bash` has already been executed before running the following steps.

## LiDAR settings (Livox Mid-360)

Configure and check the IP address (if needed):

```bash
sudo ip addr flush dev enx00e04c68030a
sudo ip addr add 192.168.1.5/24 dev enx00e04c68030a
sudo ip link set enx00e04c68030a up
ping -c 3 192.168.1.184
```

Mainline:

```bash
source ~/ws_livox/install/setup.bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

## Camera settings (GoPro Hero 13 Black)

Set up the `ros_gopro_driver`:

```bash
source ~/ws_gopro/install/setup.bash
chmod +x ~/ws_gopro/src/ros_gopro_driver/scripts/ros_gopro.py
ros2 launch ros_gopro_driver front_camera.launch.py
```

Decompress `/front_camera/image_raw/compressed` into `/image`:

```bash
ros2 run image_transport republish compressed raw \
  --ros-args \
  -r in/compressed:=/front_camera/image_raw/compressed \
  -r out:=/image
```

Verify the camera stream:

```bash
ros2 run rqt_image_view rqt_image_view
```

Check all topics:

```bash
ros2 topic list
```

## Camera calibration (fisheye)

Estimate camera intrinsics with distortions (if needed).

See files in  `CameraCalibration`.

```bash
python3 ~/CameraCalibration/topic2img.py
python3 ~/CameraCalibration/calibration_fisheye.py
```

## LiDAR-camera calibration

Record a rosbag file for calibraiton:

```bash
ros2 bag record -o livox_20250923 \
  /image /camera_info /livox/lidar /livox/imu /tf /tf_static
```

Then follow the protocal in Koide-san’s official documentation.

Check your `--camera_model`, `--camera_intrinsics` and `--camera_distortion_coeffs`:

```bash
source ~/ws_koide/install/setup.bash
ros2 run direct_visual_lidar_calibration preprocess \
  --data_path ./livox_20250923 \
  --dst_path ./livox_20250923_prep \
  --image_topic /image \
  --points_topic /livox/lidar \
  --camera_model fisheye \
  --camera_intrinsics 581.77,584.05,630.92,353.98 \
  --camera_distortion_coeffs 0.0218,0.0599,0.0096,-0.1034 \
  -v
```

For the initial guess, it is highly recommanded to use the manual option for better estimation:

```bash
ros2 run direct_visual_lidar_calibration initial_guess_manual livox_20250923_prep
```

## Calibped

Build and run Calibped:

```bash
colcon build --symlink-installsource ./install/setup.bash
source ~/ws_calibped/install/setup.bash
```

Check if the package exists:

```bash
ros2 pkg list | grep caliped
```

Run the package:

```bash
ros2 run calibped calibped_node \
        --ros-args \
        -p calib_json_path:=/HOME/livingrobot/ws_calibped/src/calibped/calibped/configs/calib.json \
        -p ped_sector_half_angle_deg:=60.0  
```

Check outputs and monitor `/calib/lidar_ped_dist`:

```bash
ros2 topic list -t
```

```bash
ros2 topic echo /calib/lidar_ped_dist
```

- Published topics:
    - `/calib/image_mask`: Masked image for the pedestrian class.
    - `/calib/lidar_ped`: Pedestrian/person points according to `/calib/image_mask`.
    - `/calib/lidar_remainder`: Remaining points from `calib/lidar_ped` after noise filtering.
    - `/calib/lidar_bg`: Background points, which is the remainder of the LiDAR scene.
    - `/calib/lidar_ped_dist`: Distance (meters) from the LiDAR sensor to the closest pedestrian point in `/calib/lidar_ped` (sensitive to image segmentation performance and camera calibration accuracy).
