"""
features:
    1) image-based person mask 
    2) lidar scene projection 
    3) person point cloud extraction (and publishes the closest distance topic too)
    This package basically read intrinsics/extrinsics from a Koide's calib.json.
    https://github.com/koide3/direct_visual_lidar_calibration
    In that file, T_lidar_camera = [tx, ty, tz, qx, qy, qz, qw] 
    describes the transform from camera to lidar.
    We intert it to get lidar -> camera, 
    since projection needs points in the camera frame.
    The camera frame here follows OpenCV's optical frame
    x: right, y: down, z: forward

commands:
    source /opt/ros/humbe/setup.bash
    source ~/ws_calibped/install/setup.bash
    ros2 run calibped calibped_node \
        --ros-args \
        -p calib_json_path:=/HOME/livingrobot/ws_calibped/src/calibped/calibped/configs/calib.json \
        -p ped_sector_half_angle_deg:=60.0      
"""
import os
import sys
import json

import cv2
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Float32
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import torch
import torch.nn.functional as F
from torchvision import transforms

from calibped.utils.common import *
# from .utils.common import *


# PIDNet path (or you can also switch other things for seg.)
current_dir = os.path.dirname(os.path.abspath(__file__))
pidnet_path = os.path.join(current_dir, 'PIDNet')
if pidnet_path not in sys.path and os.path.exists(pidnet_path):
    sys.path.insert(0, pidnet_path)
from models import pidnet  # default seg. model: pidnet
# from datasets import kusakari  # use it when kusakari


# Node
class CalibPedNode(Node):
    def __init__(self):
        super().__init__('calibped_node')

        # Parameters
        # Topics
        self.declare_parameter('input_image_topic', '/image')
        # self.declare_parameter('input_image_topic', '/front_camera/image_raw/compressed')
        self.declare_parameter('input_lidar_topic', '/livox/lidar')
        self.declare_parameter('image_mask_topic', '/calib/image_mask')
        self.declare_parameter('lidar_ped_topic', '/calib/lidar_ped')
        self.declare_parameter('lidar_remainder_topic', '/calib/lidar_remainder')  # new!
        self.declare_parameter('lidar_bg_topic', '/calib/lidar_bg')
        self.declare_parameter('lidar_ped_dist_topic', '/calib/lidar_ped_dist')

        # PIDNet
        self.declare_parameter('model_path', os.path.join(current_dir, 'PIDNet', 'pretrained_models', 'cityscapes', 'PIDNet_L_Cityscapes_test.pt'))
        # self.declare_parameter('model_path', os.path.join(current_dir, 'PIDNet', 'pretrained_models', 'kusakari', 'best_kawahara.pt'))  # when using kusakari
        self.declare_parameter('device', 'cpu')  # 'cuda' or 'cpu'
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('overlay_mask', False)
        self.declare_parameter('confidence_threshold', 0.7)

        # et cetera
        self.declare_parameter('calib_json_path', './configs/calib.json')  # e.g., /path/to/calib.json
        self.declare_parameter('ped_sector_center_deg', 0.0)  # new!
        self.declare_parameter('ped_sector_half_angle_deg', 60.0)  # new!

        # Get params
        self.input_image_topic = self.get_parameter('input_image_topic').value
        self.input_lidar_topic = self.get_parameter('input_lidar_topic').value
        self.image_mask_topic = self.get_parameter('image_mask_topic').value
        self.lidar_ped_topic = self.get_parameter('lidar_ped_topic').value
        self.lidar_remainder_topic = self.get_parameter('lidar_remainder_topic').value  # new!
        self.lidar_bg_topic = self.get_parameter('lidar_bg_topic').value
        self.lidar_ped_dist_topic = self.get_parameter('lidar_ped_dist_topic').value

        self.model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.overlay_mask = bool(self.get_parameter('overlay_mask').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)

        self.calib_json_path = self.get_parameter('calib_json_path').value
        if not self.calib_json_path:
            raise ValueError('calib_json_path is required.')
        
        self.ped_sector_center = np.deg2rad(float(self.get_parameter('ped_sector_center_deg').value))  # new!
        self.ped_sector_half = np.deg2rad(float(self.get_parameter('ped_sector_half_angle_deg').value))  # new!

        # load calib.json (both intrinsics and extrinsics)
        with open(self.calib_json_path, 'r') as f:
            cal = json.load(f)

        # Intrinsics
        # cal['camera']['intrinsics'] = [fx, fy, cx, cy]
        intr = cal['camera']['intrinsics']
        self.K = np.array([
            intr[0], 0.0, intr[2],
            0.0, intr[1], intr[3],
            0.0, 0.0, 1.0
        ], dtype=np.float64).reshape(3, 3)

        # Distortion
        # camera_model: 'fisheye' or 'pinhole'
        self.camera_model = cal['camera'].get('camera_model', 'fisheye').lower()
        d = cal['camera'].get('distortion_coeffs', [0.0, 0.0, 0.0, 0.0])
        if self.camera_model == 'fisheye':
            # [k1,k2,k3,k4] expected by cv2.fisheye
            d = (np.array(d, dtype=np.float64).flatten()[:4]).tolist() + [0.0] * max(0, 4 - len(d))
            self.D = np.array(d[:4], dtype=np.float64).reshape(4, 1)
        else:
            # pinhole(plumb_bob): [k1,k2,p1,p2,k3]
            d = (np.array(d, dtype=np.float64).flatten()[:5]).tolist() + [0.0] * max(0, 5 - len(d))
            self.D = np.array(d[:5], dtype=np.float64).flatten()

        # Extrinsics: 
        # T_lidar_camera = camera->lidar
        T = cal['results']['T_lidar_camera']
        t_cam2lid = np.array(T[0:3], dtype=np.float64)
        qx, qy, qz, qw = T[3:7]
        R_cam2lid = quat_to_rot(qx, qy, qz, qw)

        # LiDAR->camera for projection: invert
        R_lid2cam, t_lid2cam = invert_rt(R_cam2lid, t_cam2lid)
        self.R_cam_lidar = R_lid2cam
        self.t_cam_lidar = t_lid2cam

        # Model init
        self.bridge = CvBridge()
        self.model = None
        self.transform = None
        self._init_model()

        # QoS and pubs/subs
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, 
                         depth=10,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT
                         )
        self.image_mask_pub = self.create_publisher(Image, self.image_mask_topic, qos)
        self.lidar_ped_pub = self.create_publisher(PointCloud2, self.lidar_ped_topic, qos)
        self.lidar_remainder_pub = self.create_publisher(PointCloud2, self.lidar_remainder_topic, qos)
        self.lidar_bg_pub = self.create_publisher(PointCloud2, self.lidar_bg_topic,  qos)
        self.lidar_ped_dist_pub = self.create_publisher(Float32, self.lidar_ped_dist_topic, qos)

        self.sub_img = Subscriber(self, Image, self.input_image_topic, qos_profile=qos)
        self.sub_cloud = Subscriber(self, PointCloud2, self.input_lidar_topic, qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([self.sub_img, self.sub_cloud], queue_size=30, slop=0.05)
        self.sync.registerCallback(self.sync_callback)

        self.get_logger().info(f"Initialized. camera_model={self.camera_model}, using Koide calib.json for K/D/extrinsics.")


    # PIDNet
    def _init_model(self):
        try:
            device = torch.device(self.device)
            self.get_logger().info(f'Loading PIDNet model from {self.model_path}')
            tmp_dict = torch.load(self.model_path, map_location='cpu')

            if 'cityscapes' in self.model_path.lower():
                self.num_classes = 19
                self.person_class = 11
            elif 'kusakari' in self.model_path.lower():
                self.num_classes = 3
                self.person_class = 0
            else:
                raise ValueError(f'Invalid model path: {self.model_path}')

            self.model = pidnet.PIDNet(
                m=3, n=4,
                num_classes=self.num_classes,
                planes=64,
                ppm_planes=112,
                head_planes=256,
                augment=False
            ).to(device)

            model_dict = self.model.state_dict()
            seg_dict = {}
            for k, v in tmp_dict.items():
                key = k[6:]  # strip 'module.' if present
                if key in model_dict and v.shape == model_dict[key].shape:
                    seg_dict[key] = v
            self.model.load_state_dict(seg_dict, strict=False)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.height, self.width)),
                transforms.Normalize((0.469, 0.500, 0.398), (0.235, 0.238, 0.275)),
            ])
            self.get_logger().info('PIDNet model loaded.')
        except Exception as e:
            self.get_logger().error(f'PIDNet init failed: {e}')
            raise


    # Vision helpers
    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(frame_rgb)
        frame_tensor = self.transform(pil_image).unsqueeze(0)
        return frame_tensor


    def _predict_seg(self, frame_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            frame_tensor = frame_tensor.to(self.device)
            pred = self.model(frame_tensor)
            pred = F.interpolate(pred, size=frame_tensor.size()[-2:], mode='bilinear', align_corners=True)
            probs = F.softmax(pred, dim=1)
            seg = torch.argmax(pred, dim=1).cpu().numpy()[0]
            conf = torch.max(probs, dim=1)[0].cpu().numpy()[0]
            seg[conf < self.confidence_threshold] = 0
        return seg


    def _make_mask_output(self, original_bgr: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
        H, W = original_bgr.shape[:2]
        seg_resized = cv2.resize(segmentation.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        if self.overlay_mask:
            overlay = np.zeros_like(original_bgr, dtype=np.uint8)
            overlay[seg_resized == self.person_class] = [255, 255, 255]
            return cv2.addWeighted(original_bgr, 0.5, overlay, 0.5, 0)
        else:
            out = np.zeros((H, W), dtype=np.uint8)
            out[seg_resized == self.person_class] = 255
            return out


    def _project_points(self, Xc: np.ndarray) -> np.ndarray:
        # Xc: N×3 in camera optical frame
        objp = Xc.astype(np.float64).reshape(-1, 1, 3)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        if self.camera_model == 'fisheye':
            img_pts, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, self.K, self.D)
        else:
            img_pts, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.D if self.D.size > 0 else None)
        return img_pts.reshape(-1, 2)


    # Main callback for publishing mask, ped., and bg. points as output topics in real time
    def sync_callback(self, img_msg: Image, cloud_msg: PointCloud2):
        
        # 1) Image-based person mask
        # Image -> segmentation
        try:
            bgr = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')
            return

        seg = self._predict_seg(self._preprocess(bgr))
        H, W = bgr.shape[:2]
        seg_resized = cv2.resize(seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = (seg_resized == self.person_class)

        # Publish mask image
        mask_vis = self._make_mask_output(bgr, seg)
        try:
            enc = 'bgr8' if self.overlay_mask else 'mono8'
            msg = self.bridge.cv2_to_imgmsg(mask_vis, enc)
            msg.header = img_msg.header
            self.image_mask_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'publish image failed: {e}')

        # 2) LiDAR scene projection  
        # Point cloud -> numpy (keep intensity if present)
        field_names = [f.name for f in cloud_msg.fields]
        intensity_name = None
        for cand in ('intensity', 'reflectivity', 'Intensity', 'Reflectivity'):
            if cand in field_names:
                intensity_name = cand
                break

        try:
            wanted = ('x', 'y', 'z') + ((intensity_name,) if intensity_name else tuple())
            arr = point_cloud2.read_points_numpy(cloud_msg, field_names=wanted)
            pts = np.stack([arr['x'], arr['y'], arr['z']], axis=-1).astype(np.float32, copy=False)
            intens = arr[intensity_name] if intensity_name else None
        except Exception:
            pts_iter = point_cloud2.read_points(cloud_msg, field_names=("x","y","z"), skip_nans=True)
            pts = np.array([[p[0], p[1], p[2]] for p in pts_iter], dtype=np.float32)
            intens = None

        if pts.size == 0:
            self._publish_empty(cloud_msg.header)
            return
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if intens is not None:
            intens = intens[valid]
        if pts.shape[0] == 0:
            self._publish_empty(cloud_msg.header)
            return

        # lidar -> camera (optical) and z-front filter
        Xc = (self.R_cam_lidar @ pts.T).T + self.t_cam_lidar[None, :]
        z = Xc[:, 2]
        front = z > 1e-6
        
        # Old version
        # if not np.any(front):
        #     self._publish_clouds(cloud_msg.header, np.empty((0,3), np.float32), pts, None, intens, intensity_name,
        #                          (next((f.datatype for f in cloud_msg.fields if f.name == intensity_name), None) if intensity_name else None))
        #     self._publish_dist(0.0)
        #     return

        # New version
        if not np.any(front):
            self._publish_empty(cloud_msg.header)
            return

        Xc_f = Xc[front]
        idx_front = np.nonzero(front)[0]

        # 3) Person point cloud extraction 
        # Projection -> inside FOV -> mask hit test
        uv = self._project_points(Xc_f)
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        ped_keep = np.zeros_like(inside, dtype=bool)
        bg_keep  = np.zeros_like(inside, dtype=bool)
        if np.any(inside):
            u_in = u[inside]; v_in = v[inside]
            mask_hit = mask_bool[v_in, u_in]
            ped_keep[inside] = mask_hit
            bg_keep[inside]  = ~mask_hit

        ped_idx = idx_front[ped_keep]
        bg_idx  = idx_front[bg_keep]
        ped_pts_raw = pts[ped_idx, :] if ped_idx.size > 0 else np.empty((0,3), np.float32)
        bg_pts  = pts[bg_idx,  :] if bg_idx.size  > 0 else np.empty((0,3), np.float32)
        
        # new!
        def _wrap_to_pi(a):
            return (a + np.pi) % (2 * np.pi) - np.pi
        
        if ped_pts_raw.shape[0] > 0:
            yaw = np.arctan2(ped_pts_raw[:, 1], ped_pts_raw[:, 0])  # lidar xy-plane angle
            dtheta = _wrap_to_pi(yaw - self.ped_sector_center)
            in_sector = np.abs(dtheta) <= self.ped_sector_half

            ped_idx_in = ped_idx[in_sector]
            ped_idx_out = ped_idx[~in_sector]  # remainder

            # Final ped and remainder pts (before flip, LiDAR frame)
            ped_pts = pts[ped_idx_in, :] if ped_idx_in.size > 0 else np.empty((0,3), np.float32)
            remainder_pts= pts[ped_idx_out,:] if ped_idx_out.size > 0 else np.empty((0,3), np.float32)
        else:
            ped_idx_in = np.array([], dtype=np.int64)
            ped_idx_out = np.array([], dtype=np.int64)
            ped_pts = np.empty((0,3), np.float32)
            remainder_pts= np.empty((0,3), np.float32)
            
        bg_pts = pts[bg_idx,  :] if bg_idx.size  > 0 else np.empty((0,3), np.float32)
        
        ped_int = intens[ped_idx_in] if (intens is not None and ped_idx_in.size  > 0) else None
        remainder_int = intens[ped_idx_out] if (intens is not None and ped_idx_out.size > 0) else None
        bg_int = intens[bg_idx] if (intens is not None and bg_idx.size > 0) else None

        # Flip the extract point clouds by 180 degree around x (invert up/down in rviz)
        R_flip = np.array([[1.0, 0.0, 0.0],
                           [0.0,-1.0, 0.0],
                           [0.0, 0.0,-1.0]], dtype=np.float32)
        if ped_pts.shape[0] > 0:
            ped_pts = ped_pts @ R_flip.T
        if remainder_pts.shape[0] > 0:  # new!
            remainder_pts = remainder_pts @ R_flip.T
        if bg_pts.shape[0] > 0:
            bg_pts = bg_pts @ R_flip.T

        # Lidar_ped_dist
        # xy-planar range in lidar frame
        if ped_pts.shape[0] > 0:
            dxy = np.hypot(ped_pts[:, 0], ped_pts[:, 1]).astype(np.float32, copy=False)
            min_dist = float(dxy.min())
        else:
            # min_dist = 0.0
            min_dist = np.nan
        self._publish_dist(min_dist)

        # Publish both ped/ and bg. point clouds
        src_ifield = next((f for f in cloud_msg.fields if f.name == intensity_name), None)
        intensity_datatype = src_ifield.datatype if (intensity_name and src_ifield is not None) else None
        # ped_int = intens[ped_idx] if (intens is not None and ped_idx.size > 0) else None
        # bg_int = intens[bg_idx] if (intens is not None and bg_idx.size > 0) else None
        self._publish_clouds(cloud_msg.header, ped_pts, bg_pts, remainder_pts, ped_int, bg_int, remainder_int, intensity_name, intensity_datatype)


"""
3D LiDAR point cloud helpers
"""
def _np_dtype_for_pf(datatype: int):
    return {
        PointField.INT8: np.int8,
        PointField.UINT8: np.uint8,
        PointField.INT16: np.int16,
        PointField.UINT16: np.uint16,
        PointField.INT32: np.int32,
        PointField.UINT32: np.uint32,
        PointField.FLOAT32: np.float32,
        PointField.FLOAT64: np.float64,
    }.get(datatype, np.float32)


def _make_xyzi_fields(intensity_name: str, intensity_datatype: int):
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    fields.append(PointField(name=intensity_name, offset=12, datatype=intensity_datatype, count=1))
    return fields


"""
Publishing helpers
"""
def _publish_dist(self, d: float):
    dist_msg = Float32(); dist_msg.data = d
    self.lidar_ped_dist_pub.publish(dist_msg)

def _publish_empty(self, header_in):
    header = Header(); header.stamp = header_in.stamp; header.frame_id = header_in.frame_id
    empty = point_cloud2.create_cloud_xyz32(header, [])
    self.lidar_ped_pub.publish(empty)
    self.lidar_bg_pub.publish(empty)
    self.lidar_remainder_pub.publish(empty)  # new!
    self._publish_dist(0.0)

def _publish_clouds(self, header_in, ped_pts, bg_pts, remainder_pts, 
                    ped_int=None, bg_int=None, remainder_int=None, 
                    intensity_name='intensity', intensity_datatype=None):
    header = Header(); header.stamp = header_in.stamp; header.frame_id = header_in.frame_id

    def _pub(pc_pub, pts, inten):
        if pts is None or pts.shape[0] == 0:
            msg = point_cloud2.create_cloud_xyz32(header, [])
            pc_pub.publish(msg); return
        if inten is None or intensity_datatype is None or intensity_name is None:
            msg = point_cloud2.create_cloud_xyz32(header, pts.tolist())
            pc_pub.publish(msg); return
        fields = _make_xyzi_fields(intensity_name, intensity_datatype)
        np_dtype = _np_dtype_for_pf(intensity_datatype)
        inten_np = np.asarray(inten, dtype=np_dtype).reshape(-1)
        rows = zip(pts[:,0].tolist(), pts[:,1].tolist(), pts[:,2].tolist(), inten_np.tolist())
        msg = point_cloud2.create_cloud(header, fields, rows)
        pc_pub.publish(msg)

    _pub(self.lidar_ped_pub, ped_pts, ped_int)
    _pub(self.lidar_bg_pub, bg_pts, bg_int)
    _pub(self.lidar_remainder_pub, remainder_pts, remainder_int)


# Entry
def main(args=None):
    rclpy.init(args=args)
    node = CalibPedNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()