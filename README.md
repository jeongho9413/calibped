# CalibPed

* **CalibPed** is a ROS2 package for real-time 3D pedestrian segmentatoin using LiDAR-camera calibration.
* It has been tested with a **Livox Mid-360** and **GoPro Hero 13 Black** on **ROS2 Humble**.
* This repository is part of ongoing projects in our lab.


## Quick start

### LiDAR settings (Livox Mid-360)

Configure and check the IP address (if needed):

```bash
sudo ip addr flush dev enx00e04c68030a
sudo ip addr add 192.168.1.5/24 dev enx00e04c68030a
sudo ip link set enx00e04c68030a up
ping -c 3 192.168.1.184
```

Launch a livox LiDAR driver following the official documentation:

```bash
source /opt/ros/humble/setup.bash
source ~/ws_livox/install/setup.bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

### Camera settings (GoPro Hero 13 Black)

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

### Camera calibration

Estimate the camera intrinsics and distortion coefficients.
See the instructions on [this page](https://github.com/jeongho9413/camera-calibration). 


### LiDAR-camera calibration

Record a rosbag file for calibraiton:

```bash
ros2 bag record -o livox_20250923 \
  /image /camera_info /livox/lidar /livox/imu /tf /tf_static
```

Then follow the protocal described in [direct_visual_lidar_calibration](https://koide3.github.io/direct_visual_lidar_calibration/).

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

For the initial guess, it is highly recommanded to use the manual option for a more accurate estimation:

```bash
ros2 run direct_visual_lidar_calibration initial_guess_manual livox_20250923_prep
```

### CalibPed

Build and run CalibPed:

```bash
colcon build --symlink-install
source ./install/setup.bash
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
        -p calib_json_path:=./configs/calib.json \
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
    - `/calib/lidar_ped_dist`: Distance (meters) from the LiDAR sensor to the closest detected pedestrian point in `/calib/lidar_ped` (sensitive to segmentation performance and camera calibration accuracy).


## Acknowledgements
* The image-based segmentation is based on [`XuJiacong/PIDNet`](https://github.com/XuJiacong/PIDNet).
* The LiDAR-camera calibration is based on [`koide3/direct_visual_lidar_calibration`](https://github.com/koide3/direct_visual_lidar_calibration/tree/main).
