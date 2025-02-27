# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import random
import numpy as np
from typing import Tuple
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.clock import Clock

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class DebugNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # declare params
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.RELIABLE)
        self.declare_parameter("undistort", True)
        self.declare_parameter("scale", 0.5)

        self.declare_parameter("fc_input_topic", "/perception/camera_front_center/image")
        self.declare_parameter("rc_input_topic", "/perception/camera_rear_center/image")
        self.declare_parameter("rs_input_topic", "/perception/camera_right_side/image")
        self.declare_parameter("ls_input_topic", "/perception/camera_left_side/image")
        self.declare_parameter("fr_input_topic", "/perception/camera_front_right/image")
        self.declare_parameter("fl_input_topic", "/perception/camera_front_left/image")

        self.declare_parameter("fc_detection_topic", "/perception/post/camera_front_center/detections")
        self.declare_parameter("rc_detection_topic", "/perception/post/camera_rear_center/detections")
        self.declare_parameter("rs_detection_topic", "/perception/post/camera_right_side/detections")
        self.declare_parameter("ls_detection_topic", "/perception/post/camera_left_side/detections")
        self.declare_parameter("fr_detection_topic", "/perception/post/camera_front_right/detections")
        self.declare_parameter("fl_detection_topic", "/perception/post/camera_front_left/detections")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        # get parameters
        self.fc_input_topic = self.get_parameter("fc_input_topic").value
        self.rc_input_topic = self.get_parameter("rc_input_topic").value
        self.rs_input_topic = self.get_parameter("rs_input_topic").value
        self.ls_input_topic = self.get_parameter("ls_input_topic").value
        self.fr_input_topic = self.get_parameter("fr_input_topic").value
        self.fl_input_topic = self.get_parameter("fl_input_topic").value

        self.camera_topics = {
            'camera_front_center' : self.fc_input_topic,
            'camera_rear_center' : self.rc_input_topic,
            'camera_right_side' : self.rs_input_topic,
            'camera_left_side' : self.ls_input_topic,
            'camera_front_right' : self.fr_input_topic,
            'camera_front_left' : self.fl_input_topic,
        }

        self.fc_detection_topic = self.get_parameter("fc_detection_topic").value
        self.rc_detection_topic = self.get_parameter("rc_detection_topic").value
        self.rs_detection_topic = self.get_parameter("rs_detection_topic").value
        self.ls_detection_topic = self.get_parameter("ls_detection_topic").value
        self.fr_detection_topic = self.get_parameter("fr_detection_topic").value
        self.fl_detection_topic = self.get_parameter("fl_detection_topic").value

        self.undistort = self.get_parameter("undistort").value
        self.scale = self.get_parameter("scale").value

        # pubs
        self.fc_dbg_pub = self.create_publisher(Image, "/perception/post/camera_front_center/dbg_image", 10)
        self.fc_bb_markers_pub = self.create_publisher(MarkerArray, "/perception/post/camera_front_center/dgb_bb_markers", 10)

        self.rc_dbg_pub = self.create_publisher(Image, "/perception/post/camera_rear_center/dbg_image", 10)
        self.rc_bb_markers_pub = self.create_publisher(MarkerArray, "/perception/post/camera_rear_center/dgb_bb_markers", 10)

        self.rs_dbg_pub = self.create_publisher(Image, "/perception/post/camera_right_side/dbg_image", 10)
        self.rs_bb_markers_pub = self.create_publisher(MarkerArray, "/perception/post/camera_right_side/dgb_bb_markers", 10)

        self.ls_dbg_pub = self.create_publisher(Image, "/perception/post/camera_left_side/dbg_image", 10)
        self.ls_bb_markers_pub = self.create_publisher(MarkerArray, "/perception/post/camera_left_side/dgb_bb_markers", 10)

        self.fr_dbg_pub = self.create_publisher(Image, "/perception/post/camera_front_right/dbg_image", 10)
        self.fr_bb_markers_pub = self.create_publisher(MarkerArray, "/perception/post/camera_front_right/dgb_bb_markers", 10)

        self.fl_dbg_pub = self.create_publisher(Image, "/perception/post/camera_front_left/dbg_image", 10)
        self.fl_bb_markers_pub = self.create_publisher(MarkerArray, "/perception/post/camera_front_left/dgb_bb_markers", 10)
        
        # get flags
        self.cam_info_done = 0
        self.cam_size_done = 0

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")
       
        # Camera info topic subscribers 
        self.fc_cam_info_sub = self.create_subscription(
            CameraInfo, "/perception/test/camera_front_center/camera_info", self.get_camera_info, 10
        )
        self.rc_cam_info_sub = self.create_subscription(
            CameraInfo, "/perception/test/camera_rear_center/camera_info", self.get_camera_info, 10
        )
        self.rs_cam_info_sub = self.create_subscription(
            CameraInfo, "/perception/test/camera_right_side/camera_info", self.get_camera_info, 10
        )
        self.ls_cam_info_sub = self.create_subscription(
            CameraInfo, "/perception/test/camera_left_side/camera_info", self.get_camera_info, 10
        )
        self.fr_cam_info_sub = self.create_subscription(
            CameraInfo, "/perception/test/camera_front_right/camera_info", self.get_camera_info, 10
        )
        self.fl_cam_info_sub = self.create_subscription(
            CameraInfo, "/perception/test/camera_front_left/camera_info", self.get_camera_info, 10
        )

        # Create only active camera subscribers
        self.subscribers = []
        self.active_cam_names = []
        self.active_detections = []
        self.synchronizers = []

        for name, topic in self.camera_topics.items():
            if self.is_topic_active(topic):  # Only subscribe if the camera topic is active
                cam_sub = message_filters.Subscriber(self, Image, topic, qos_profile=self.image_qos_profile)
                self.subscribers.append(cam_sub)
                self.active_cam_names.append(name)
                self.get_logger().info(f"Subscribed to {topic}")

                # Subscribe to the corresponding detection topic
                detection_topic = f"/perception/post/{name}/detections"
                det_sub = message_filters.Subscriber(self, DetectionArray, detection_topic, qos_profile=10)
                self.active_detections.append((cam_sub, det_sub))

        self.get_logger().info(f"Subscribed cameras: {self.active_cam_names}")

        # Create synchronizers for only active camera-detection pairs
        for cam_sub, det_sub in self.active_detections:
            sync = message_filters.ApproximateTimeSynchronizer((cam_sub, det_sub), 10, 0.5)
            sync.registerCallback(self.detections_cb)
            self.get_logger().info(f"Synchronizer created for {cam_sub} and {det_sub}")
            self.synchronizers.append(sync)

        self.get_logger().info(f"Active synchronizers: {len(self.synchronizers)}")



        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS
    
    def is_topic_active(self, topic_name):
        # Check if a topic is actively being published 
        topic_list = [topic for topic, _ in self.get_topic_names_and_types()]
        
        return topic_name in topic_list  # Check exact match

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        # destroy cam subs 
        self.destroy_subscription(self.fc_image_sub)
        self.fc_image_sub = None
        self.destroy_subscription(self.rc_image_sub)
        self.rc_image_sub = None
        self.destroy_subscription(self.rs_image_sub)
        self.rs_image_sub = None
        self.destroy_subscription(self.ls_image_sub)
        self.ls_image_sub = None
        self.destroy_subscription(self.fr_image_sub)
        self.fr_image_sub = None
        self.destroy_subscription(self.fl_image_sub)
        self.fl_image_sub = None

        # destroy detection subs
        self.destroy_subscription(self.fc_detections_sub)
        self.fc_detections_sub = None
        self.destroy_subscription(self.rc_detections_sub)
        self.rc_detections_sub = None
        self.destroy_subscription(self.rs_detections_sub)
        self.rs_detections_sub = None
        self.destroy_subscription(self.ls_detections_sub)
        self.ls_detections_sub = None
        self.destroy_subscription(self.fr_detections_sub)
        self.fr_detections_sub = None
        self.destroy_subscription(self.fl_detections_sub)
        self.fl_detections_sub = None

        # destroy cam info subs
        self.destroy_subscription(self.fc_cam_info_sub)
        self.fc_cam_info_sub = None
        self.destroy_subscription(self.rc_cam_info_sub)
        self.rc_cam_info_sub = None
        self.destroy_subscription(self.rs_cam_info_sub)
        self.rs_cam_info_sub = None
        self.destroy_subscription(self.ls_cam_info_sub)
        self.ls_cam_info_sub = None
        self.destroy_subscription(self.fr_cam_info_sub)
        self.fr_cam_info_sub = None
        self.destroy_subscription(self.fl_cam_info_sub)
        self.fl_cam_info_sub = None

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self.fc_dbg_pub)
        self.destroy_publisher(self.rc_dbg_pub)
        self.destroy_publisher(self.rs_dbg_pub)
        self.destroy_publisher(self.ls_dbg_pub)
        self.destroy_publisher(self.fr_dbg_pub)
        self.destroy_publisher(self.fl_dbg_pub)

        self.destroy_publisher(self.fc_bb_markers_pub)
        self.destroy_publisher(self.rc_bb_markers_pub)
        self.destroy_publisher(self.rs_bb_markers_pub)
        self.destroy_publisher(self.ls_bb_markers_pub)
        self.destroy_publisher(self.fr_bb_markers_pub)
        self.destroy_publisher(self.fl_bb_markers_pub)
        
        # self.destroy_publisher(self._kp_markers_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def draw_box(
        self, cv_image: np.ndarray, detection: Detection, color: Tuple[int]
    ) -> np.ndarray:

        # get detection info
        class_name = detection.class_name
        score = detection.score
        box_msg: BoundingBox2D = detection.bbox
        track_id = detection.id

        min_pt = (
            round(box_msg.center.position.x - box_msg.size.x / 2.0),
            round(box_msg.center.position.y - box_msg.size.y / 2.0),
        )
        max_pt = (
            round(box_msg.center.position.x + box_msg.size.x / 2.0),
            round(box_msg.center.position.y + box_msg.size.y / 2.0),
        )

        # define the four corners of the rectangle
        rect_pts = np.array(
            [
                [min_pt[0], min_pt[1]],
                [max_pt[0], min_pt[1]],
                [max_pt[0], max_pt[1]],
                [min_pt[0], max_pt[1]],
            ]
        )

        # calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            (box_msg.center.position.x, box_msg.center.position.y),
            -np.rad2deg(box_msg.center.theta),
            1.0,
        )

        # rotate the corners of the rectangle
        rect_pts = np.int0(cv2.transform(np.array([rect_pts]), rotation_matrix)[0])

        # Draw the rotated rectangle
        for i in range(4):
            pt1 = tuple(rect_pts[i])
            pt2 = tuple(rect_pts[(i + 1) % 4])
            cv2.line(cv_image, pt1, pt2, color, 2)

        # write text
        label = f"{class_name}"
        label += f" ({track_id})" if track_id else ""
        label += " ({:.3f})".format(score)
        pos = (min_pt[0] + 5, min_pt[1] + 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font, 1, color, 1, cv2.LINE_AA)

        return cv_image

    def draw_mask(
        self, cv_image: np.ndarray, detection: Detection, color: Tuple[int]
    ) -> np.ndarray:

        mask_msg = detection.mask
        mask_array = np.array([[int(ele.x), int(ele.y)] for ele in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            cv_image = cv2.polylines(
                cv_image,
                [mask_array],
                isClosed=True,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        return cv_image

    def draw_keypoints(self, cv_image: np.ndarray, detection: Detection) -> np.ndarray:

        keypoints_msg = detection.keypoints

        cv_image = np.ascontiguousarray(cv_image)  # Fix non-contiguous memory
        ann = Annotator(cv_image)

        kp: KeyPoint2D
        for kp in keypoints_msg.data:
            color_k = (
                [int(x) for x in ann.kpt_color[kp.id - 1]]
                if len(keypoints_msg.data) == 17
                else colors(kp.id - 1)
            )

            cv2.circle(
                cv_image,
                (int(kp.point.x), int(kp.point.y)),
                5,
                color_k,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                cv_image,
                str(kp.id),
                (int(kp.point.x), int(kp.point.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_k,
                1,
                cv2.LINE_AA,
            )

        def get_pk_pose(kp_id: int) -> Tuple[int]:
            for kp in keypoints_msg.data:
                if kp.id == kp_id:
                    return (int(kp.point.x), int(kp.point.y))
            return None

        for i, sk in enumerate(ann.skeleton):
            kp1_pos = get_pk_pose(sk[0])
            kp2_pos = get_pk_pose(sk[1])

            if kp1_pos is not None and kp2_pos is not None:
                cv2.line(
                    cv_image,
                    kp1_pos,
                    kp2_pos,
                    [int(x) for x in ann.limb_color[i]],
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        return cv_image

    def create_bb_marker(self, detection: Detection, color: Tuple[int]) -> Marker:

        bbox3d = detection.bbox3d

        marker = Marker()
        marker.header.frame_id = bbox3d.frame_id

        marker.ns = "yolo_3d"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = bbox3d.center.position.x
        marker.pose.position.y = bbox3d.center.position.y
        marker.pose.position.z = bbox3d.center.position.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = bbox3d.size.x
        marker.scale.y = bbox3d.size.y
        marker.scale.z = bbox3d.size.z

        marker.color.r = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.b = color[2] / 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = detection.class_name

        return marker

    def create_kp_marker(self, keypoint: KeyPoint3D) -> Marker:

        marker = Marker()

        marker.ns = "yolo_3d"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = keypoint.point.x
        marker.pose.position.y = keypoint.point.y
        marker.pose.position.z = keypoint.point.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = (1.0 - keypoint.score) * 255.0
        marker.color.g = 0.0
        marker.color.b = keypoint.score * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = str(keypoint.id)

        return marker

    def get_camera_info(self, cam_info: CameraInfo):
    
        # decide which camera you need the cam_info for
        self.get_logger().info(f"[{self.get_name()}] getting {cam_info.header.frame_id} camera info")
        self.get_logger().info(f"[{self.get_name()}] cam info is at {self.cam_info_done+1}")

        # kill the subscriber
        if cam_info.header.frame_id == "camera_front_center" and self.fc_cam_info_sub:  # front center camera
            self.fc_k_mtx = np.array(cam_info.k).reshape((3, 3))  # Convert to 3x3 matrix Camera intrinsic parameters 
            self.fc_d_mtx = np.array(cam_info.d)                  # Convert distortion coefficients to NumPy array Distortion Coefficients
            self.destroy_subscription(self.fc_cam_info_sub)
            self.fc_cam_info_sub = None
            self.cam_info_done += 1
        elif cam_info.header.frame_id == "camera_rear_center" and self.rc_cam_info_sub: # rear center camera
            self.rc_k_mtx = np.array(cam_info.k).reshape((3, 3))  # Convert to 3x3 matrix Camera intrinsic parameters 
            self.rc_d_mtx = np.array(cam_info.d)                  # Convert distortion coefficients to NumPy array Distortion Coefficients
            self.destroy_subscription(self.rc_cam_info_sub)
            self.rc_cam_info_sub = None
            self.cam_info_done += 1
        elif cam_info.header.frame_id == "camera_right_side" and self.rs_cam_info_sub: # right side camera
            self.rs_k_mtx = np.array(cam_info.k).reshape((3, 3))  # Convert to 3x3 matrix Camera intrinsic parameters 
            self.rs_d_mtx = np.array(cam_info.d)                  # Convert distortion coefficients to NumPy array Distortion Coefficients
            self.destroy_subscription(self.rs_cam_info_sub)
            self.rs_cam_info_sub = None
            self.cam_info_done += 1
        elif cam_info.header.frame_id == "camera_left_side" and self.ls_cam_info_sub: # left side camera
            self.ls_k_mtx = np.array(cam_info.k).reshape((3, 3))  # Convert to 3x3 matrix Camera intrinsic parameters 
            self.ls_d_mtx = np.array(cam_info.d)                  # Convert distortion coefficients to NumPy array Distortion Coefficients
            self.destroy_subscription(self.ls_cam_info_sub)
            self.ls_cam_info_sub = None
            self.cam_info_done += 1   
        elif cam_info.header.frame_id == "camera_front_right" and self.fr_cam_info_sub: # front right camera
            self.fr_k_mtx = np.array(cam_info.k).reshape((3, 3))  # Convert to 3x3 matrix Camera intrinsic parameters 
            self.fr_d_mtx = np.array(cam_info.d)                  # Convert distortion coefficients to NumPy array Distortion Coefficients
            self.destroy_subscription(self.fr_cam_info_sub)
            self.fr_cam_info_sub = None
            self.cam_info_done += 1
        elif cam_info.header.frame_id == "camera_front_left" and self.fl_cam_info_sub: # front left camera
            self.fl_k_mtx = np.array(cam_info.k).reshape((3, 3))  # Convert to 3x3 matrix Camera intrinsic parameters 
            self.fl_d_mtx = np.array(cam_info.d)                  # Convert distortion coefficients to NumPy array Distortion Coefficients
            self.destroy_subscription(self.fl_cam_info_sub)
            self.fl_cam_info_sub = None
            self.cam_info_done += 1

    def get_dst_map(self, msg: Image, cv_image: np.ndarray):   # function to get cam size, optimal new matrix, and undist map
 
        # Get image size
        h, w = cv_image.shape[:2]  # Assuming cv_image is the input image
        R = np.eye(3, dtype=np.float32)  # Rectification matrix (Identity if not stereo)

        if msg.header.frame_id == "camera_front_center":
            self.fc_new_camera_mtx, self.fc_roi = cv2.getOptimalNewCameraMatrix(self.fc_k_mtx, self.fc_d_mtx, (w, h), 0, (w, h))   
            self.fc_map1, self.fc_map2 = cv2.initUndistortRectifyMap(self.fc_k_mtx, self.fc_d_mtx, R, self.fc_new_camera_mtx, (w,h), cv2.CV_32FC1)     
            self.cam_size_done += 1
        elif msg.header.frame_id == "camera_rear_center":
            self.rc_new_camera_mtx, self.rc_roi = cv2.getOptimalNewCameraMatrix(self.rc_k_mtx, self.rc_d_mtx, (w, h), 0, (w, h))
            self.rc_map1, self.rc_map2 = cv2.initUndistortRectifyMap(self.rc_k_mtx, self.rc_d_mtx, R, self.rc_new_camera_mtx, (w,h), cv2.CV_32FC1)     
            self.cam_size_done += 1
        elif msg.header.frame_id == "camera_right_side":
            self.rs_new_camera_mtx, self.rs_roi = cv2.getOptimalNewCameraMatrix(self.rs_k_mtx, self.rs_d_mtx, (w, h), 0, (w, h))
            self.rs_map1, self.rs_map2 = cv2.initUndistortRectifyMap(self.rs_k_mtx, self.rs_d_mtx, R,  self.rs_new_camera_mtx, (w,h), cv2.CV_32FC1)     
            self.cam_size_done += 1
        elif msg.header.frame_id == "camera_left_side":
            self.ls_new_camera_mtx, self.ls_roi = cv2.getOptimalNewCameraMatrix(self.ls_k_mtx, self.ls_d_mtx, (w, h), 0, (w, h))
            self.ls_map1, self.ls_map2 = cv2.initUndistortRectifyMap(self.ls_k_mtx, self.ls_d_mtx, R, self.ls_new_camera_mtx, (w,h), cv2.CV_32FC1)     
            self.cam_size_done += 1
        elif msg.header.frame_id == "camera_front_right":
            self.fr_new_camera_mtx, self.fr_roi = cv2.getOptimalNewCameraMatrix(self.fr_k_mtx, self.fr_d_mtx, (w, h), 0, (w, h))
            self.fr_map1, self.fr_map2 = cv2.initUndistortRectifyMap(self.fr_k_mtx, self.fr_d_mtx, R, self.fr_new_camera_mtx, (w,h), cv2.CV_32FC1)     #TODO: Add stereo rectification here    
            self.cam_size_done += 1
        elif msg.header.frame_id == "camera_front_left":
            self.fl_new_camera_mtx, self.fl_roi = cv2.getOptimalNewCameraMatrix(self.fl_k_mtx, self.fl_d_mtx, (w, h), 0, (w, h))
            self.fl_map1, self.fl_map2 = cv2.initUndistortRectifyMap(self.fl_k_mtx, self.fl_d_mtx, R, self.fl_new_camera_mtx, (w,h), cv2.CV_32FC1)     #TODO: Add stereo rectification here
            self.cam_size_done += 1

    def undistort_image(self, msg: Image, cv_image: np.ndarray):

        if self.cam_size_done < 6: # get optimal camera matrix only once
                self.get_dst_map(msg, cv_image)
                self.get_logger().info(f"cam_size_done =  {self.cam_size_done}")

        else:
            # get camera info
            if msg.header.frame_id == "camera_front_center": 
                roi = self.fc_roi
                map1, map2 = self.fc_map1, self.fc_map2
            if msg.header.frame_id == "camera_rear_center": 
                roi = self.rc_roi
                map1, map2 = self.rc_map1, self.rc_map2
            if msg.header.frame_id == "camera_right_side": 
                roi = self.rs_roi
                map1, map2 = self.rs_map1, self.rs_map2
            if msg.header.frame_id == "camera_left_side": 
                roi = self.ls_roi
                map1, map2 = self.ls_map1, self.ls_map2
            if msg.header.frame_id == "camera_front_right": 
                roi = self.fr_roi
                map1, map2 = self.fr_map1, self.fr_map2
            if msg.header.frame_id == "camera_front_left": 
                roi = self.fl_roi
                map1, map2 = self.fl_map1, self.fl_map2

            # undistort the image before running YOLO
            dst = cv2.remap(cv_image, map1, map2, cv2.INTER_NEAREST)

            # crop the image
            x, y, w, h = roi 
            un_dist_image = dst[y:y+h,x:x+w]

            return un_dist_image

    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray) -> None:

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
        clock = Clock()
        start_time = clock.now()

        # downsample image
        new_size = (int(cv_image.shape[1] * self.scale), int(cv_image.shape[0] * self.scale))
        cv_image = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_NEAREST)

        end_time = clock.now()
        duration_ns = end_time.nanoseconds - start_time.nanoseconds
        duration_s = duration_ns / 1e9  # Convert nanoseconds to seconds
        
        # if unndistort is enabled and cam info is stored, then undistort the image
        if self.undistort and self.cam_info_done == 6:
            un_dist_image = self.undistort_image(img_msg, cv_image)
            if self.cam_size_done == 7:
                cv_image = un_dist_image

        bb_marker_array = MarkerArray()
        kp_marker_array = MarkerArray()

        detection: Detection
        for detection in detection_msg.detections:

            # random color
            class_name = detection.class_name

            if class_name not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[class_name] = (r, g, b)

            color = self._class_to_color[class_name]

            cv_image = self.draw_box(cv_image, detection, color)
            cv_image = self.draw_mask(cv_image, detection, color)
            cv_image = self.draw_keypoints(cv_image, detection)

            if detection.bbox3d.frame_id:
                marker = self.create_bb_marker(detection, color)
                marker.header.stamp = img_msg.header.stamp
                marker.id = len(bb_marker_array.markers)
                bb_marker_array.markers.append(marker)

            if detection.keypoints3d.frame_id:
                for kp in detection.keypoints3d.data:
                    marker = self.create_kp_marker(kp)
                    marker.header.frame_id = detection.keypoints3d.frame_id
                    marker.header.stamp = img_msg.header.stamp
                    marker.id = len(kp_marker_array.markers)
                    kp_marker_array.markers.append(marker)
        
        # publish dbg image
        if img_msg.header.frame_id == "camera_front_center":
            self.fc_dbg_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            )
            self.fc_bb_markers_pub.publish(bb_marker_array)
            # self._kp_markers_pub.publish(kp_marker_array)
        elif img_msg.header.frame_id == "camera_rear_center":
            self.rc_dbg_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            )
            self.rc_bb_markers_pub.publish(bb_marker_array)
            # self._kp_markers_pub.publish(kp_marker_array)
        elif img_msg.header.frame_id == "camera_right_side":
            self.rs_dbg_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            )
            self.rs_bb_markers_pub.publish(bb_marker_array)
            # self._kp_markers_pub.publish(kp_marker_array)
        elif img_msg.header.frame_id == "camera_left_side":
            self.ls_dbg_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            )
            self.ls_bb_markers_pub.publish(bb_marker_array)
            # self._kp_markers_pub.publish(kp_marker_array)
        elif img_msg.header.frame_id == "camera_front_right":
            self.fr_dbg_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            )
            self.fr_bb_markers_pub.publish(bb_marker_array)
            # self._kp_markers_pub.publish(kp_marker_array)
        elif img_msg.header.frame_id == "camera_front_left":
            self.fl_dbg_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            )
            self.fl_bb_markers_pub.publish(bb_marker_array)
            # self._kp_markers_pub.publish(kp_marker_array)


def main():
    rclpy.init()
    node = DebugNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
