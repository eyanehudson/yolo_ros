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


# Naming convention for cameras in the code:
# fc = front center camera
# rc = rear center camera
# rs = right side camera
# ls = left side camera
# fr = front right camera
# fl = front left camera

import cv2
import os
import sys


from typing import List, Dict
from cv_bridge import CvBridge

import numpy as np
import time

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.clock import Clock
from rclpy.executors import MultiThreadedExecutor


import torch
from ultralytics import YOLO, NAS, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
import message_filters
from message_filters import Subscriber, ApproximateTimeSynchronizer



from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses


# import custom msg type
from blackandgold_msgs.msg import CameraBoundingBoxes, CameraBox


from ament_index_python.packages import get_package_share_directory
file_location = get_package_share_directory('yolo_ros')
sys.path.insert(0, os.path.join(file_location))

class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        # declare all params 
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov11_best_3JAN.pt")
        self.declare_parameter("device", "cuda:0")

        self.declare_parameter("threshold", 0.75)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 3)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)

        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.RELIABLE)

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
        
        self.declare_parameter("undistort", True)
        self.declare_parameter("scale", 0.5)
        self.declare_parameter("frequency", 10)

        self.type_to_model = {"YOLO": YOLO, "NAS": NAS, "World": YOLOWorld}


    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # get all params from params.yaml file
        self.model_type = self.get_parameter("model_type").value
        self.model_ = self.get_parameter("model").value # this is the name of the .pt model
        self.model = os.path.join(file_location, 'config', self.model_) # this is the full path of the .pt model
        
        self.device = self.get_parameter("device").value
        self.threshold = self.get_parameter("threshold").value
        self.iou = self.get_parameter("iou").value
        self.imgsz_height = self.get_parameter("imgsz_height").value
        self.imgsz_width = self.get_parameter("imgsz_width").value
        self.half = self.get_parameter("half").value
        self.max_det = self.get_parameter("max_det").value
        self.augment = self.get_parameter("augment").value
        self.agnostic_nms = self.get_parameter("agnostic_nms").value
        self.retina_masks = self.get_parameter("retina_masks").value
        self.enable = self.get_parameter("enable").value
        self.reliability = self.get_parameter("image_reliability").value

        self.fc_input_topic = self.get_parameter("fc_input_topic").value
        self.rc_input_topic = self.get_parameter("rc_input_topic").value
        self.rs_input_topic = self.get_parameter("rs_input_topic").value
        self.ls_input_topic = self.get_parameter("ls_input_topic").value
        self.fr_input_topic = self.get_parameter("fr_input_topic").value
        self.fl_input_topic = self.get_parameter("fl_input_topic").value

        self.camera_topics = {
            'fc' : self.fc_input_topic,
            'rc' : self.rc_input_topic,
            'rs' : self.rs_input_topic,
            'ls' : self.ls_input_topic,
            'fr' : self.fr_input_topic,
            'fl' : self.fl_input_topic,
        }

        self.fc_detection_topic = self.get_parameter("fc_detection_topic").value
        self.rc_detection_topic = self.get_parameter("rc_detection_topic").value
        self.rs_detection_topic = self.get_parameter("rs_detection_topic").value
        self.ls_detection_topic = self.get_parameter("ls_detection_topic").value
        self.fr_detection_topic = self.get_parameter("fr_detection_topic").value
        self.fl_detection_topic = self.get_parameter("fl_detection_topic").value

        self.undistort = self.get_parameter("undistort").value
        self.scale = self.get_parameter("scale").value
        self.frequency = self.get_parameter("frequency").value

        # set image qos
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        # detection publisher
        self.fc_pub = self.create_lifecycle_publisher(DetectionArray, self.fc_detection_topic, 10)
        self.rc_pub = self.create_lifecycle_publisher(DetectionArray, self.rc_detection_topic, 10)
        self.rs_pub = self.create_lifecycle_publisher(DetectionArray, self.rs_detection_topic, 10)
        self.ls_pub = self.create_lifecycle_publisher(DetectionArray, self.ls_detection_topic, 10)
        self.fr_pub = self.create_lifecycle_publisher(DetectionArray, self.fr_detection_topic, 10)
        self.fl_pub = self.create_lifecycle_publisher(DetectionArray, self.fl_detection_topic, 10)

        # set flags
        self.cam_info_done = 0
        self.cam_size_done = 0
        self.last_callback_time = 0  # Track last execution time
        self.delay = 0.99 * (1 / self.frequency) # .99 to make the function run slightly faster than the desired frequency to account for loop delay

        self.clock = Clock()
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        if "cuda" in self.device:
            self.get_logger().info(f"Using [{self.device}] as gpu")
        elif "cpu" in self.device:
            self.get_logger().info(f"Using [{self.device}] as cpu")
            self.device = torch.device("cpu")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
            self.get_logger().info(f"Using [{self.model_}]")
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR

        try:
            self.get_logger().info("Trying to fuse model...")
            self.yolo.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fuse: {e}")

        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        if isinstance(self.yolo, YOLOWorld):
            self._set_classes_srv = self.create_service(
                SetClasses, "set_classes", self.set_classes_cb
            )
        
        # Create only active camera subscribers
        self.subscribers = []
        self.active_cam_names = []
        for name, topic in self.camera_topics.items():
            if self.is_topic_active(topic):  # Only subscribe if the topic is active
                sub = message_filters.Subscriber(self, Image, topic, qos_profile=self.image_qos_profile)
                self.subscribers.append(sub)
                self.active_cam_names.append(name)
                self.get_logger().info(f"Subscribed to {topic}")
        self.get_logger().info(f"Subscribed cameras are {self.active_cam_names}")


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

        # Synchronization only when at least one camera is active
        if len(self.subscribers) > 0:
            self.sync = message_filters.ApproximateTimeSynchronizer(self.subscribers, queue_size=10, slop=0.2)
            self.sync.registerCallback(self.wrapped_callback)
            if len(self.subscribers) < 6:
                self.get_logger().warn(f"[{self.get_name()}] ONLY {len(self.subscribers)} CAMERAS ARE BEING USED: {self.active_cam_names}")

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        if isinstance(self.yolo, YOLOWorld):
            self.destroy_service(self._set_classes_srv)
            self._set_classes_srv = None

        # destroy cam subs
        self.destroy_subscription(self.fc_sub)
        self.fc_sub = None
        self.destroy_subscription(self.rc_sub)
        self.rc_sub = None
        self.destroy_subscription(self.rs_sub)
        self.rs_sub = None
        self.destroy_subscription(self.ls_sub)
        self.ls_sub = None
        self.destroy_subscription(self.fr_sub)
        self.fr_sub = None
        self.destroy_subscription(self.fl_sub)
        self.fl_sub = None

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

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self.fc_pub)
        self.destroy_publisher(self.rc_pub)
        self.destroy_publisher(self.rs_pub)
        self.destroy_publisher(self.ls_pub)
        self.destroy_publisher(self.fr_pub)
        self.destroy_publisher(self.fl_pub)

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def is_topic_active(self, topic_name):
        # Check if a topic is actively being published 
        topic_list = [topic for topic, _ in self.get_topic_names_and_types()]
        
        return topic_name in topic_list  # Check exact match

    
    def wrapped_callback(self, *images):
        # self.start_time = self.clock.now()

        current_time = time.time()
        time_since_last_callback = current_time - self.last_callback_time
        self.get_logger().info(f"callback at {1/(time_since_last_callback):.3f} Hz")

        while time_since_last_callback < self.delay: #10hz
            time.sleep(self.delay - time_since_last_callback)
            current_time = time.time()  # Update time after sleeping
            time_since_last_callback = current_time - self.last_callback_time  # Recalculate interval
        
        # Convert positional arguments (*args) into named keyword arguments (**kwargs)
        self.get_logger().info(f"running callback at {1/(time_since_last_callback):.3f} hz")

        kwargs = {name: img for name, img in zip(self.active_cam_names, images)}
        self.image_cb(**kwargs)  # Call image_cb with named arguments
        # self.get_logger().info(f"finished at {(time.time() - self.last_callback_time):.3f} sec")

        self.last_callback_time = current_time

    def enable_cb(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response

    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:

                msg = BoundingBox2D()

                # get boxes values
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()

                # get boxes values
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

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
        h, w = cv_image.shape[:2]  # Assuming cv_image is cv_bridthe input image
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
            dst = cv2.remap(cv_image, map1, map2, cv2.INTER_NEAREST) # TODO: If bad quality, change to INTER_LINEAR

            # crop the image
            x, y, w, h = roi 
            un_dist_image = dst[y:y+h,x:x+w]
            
            return un_dist_image


    def image_cb(self, **kwargs) -> None:

        if self.enable:
            self.start_time = self.clock.now()
            end_time = self.clock.now()
            duration_ns = end_time.nanoseconds - self.start_time.nanoseconds
            duration_ms = duration_ns / 1e6  # Convert nanoseconds to seconds
            # self.get_logger().info(f"Starting image_cb: {duration_ms:.3f} ms")

            
            # TODO: For debugging
            # self.get_logger().info(f"[{self.get_name()}] Start processing {msg.header.frame_id} at {start_time.nanoseconds} nanoseconds")

            # Mapping of frame IDs to publisher objects
            camera_publishers = {
                "camera_front_center": self.fc_pub,
                "camera_rear_center": self.rc_pub,
                "camera_right_side": self.rs_pub,
                "camera_left_side": self.ls_pub,
                "camera_front_right": self.fr_pub,
                "camera_front_left": self.fl_pub,
            }
            final_pub = []
            final_det = []
            batch = []
            image_batch = []
            msg_batch = []
            
            for camera_name, image in kwargs.items():
                self.get_logger().info(f"Processing image from {camera_name}")
                end_time = self.clock.now()
                duration_ns = end_time.nanoseconds - self.start_time.nanoseconds
                duration_ms = duration_ns / 1e6  # Convert nanoseconds to seconds
                self.get_logger().info(f"entering for loop Function duration: {duration_ms:.3f} ms")

                msg = kwargs[camera_name]

                # convert to cv_image
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

                # downsample image
                new_size = (int(cv_image.shape[1] * self.scale), int(cv_image.shape[0] * self.scale))
                cv_image = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_NEAREST)

                image_batch.append(cv_image) # store cv_image
                msg_batch.append(msg) # store ros2 msg

            end_time = self.clock.now()
            duration_ns = end_time.nanoseconds - self.start_time.nanoseconds
            duration_ms = duration_ns / 1e6  # Convert nanoseconds to seconds
            self.get_logger().info(f"cv_image convert and downsample Function duration: {duration_ms:.3f} ms")

            for i in range(len(image_batch)):

                # if unndistort is enabled and cam info is stored, then undistort the image
                if self.undistort and self.cam_info_done == 6:
                        
                    un_dist_image = self.undistort_image(msg_batch[i], image_batch[i])
                    
                    batch.append(un_dist_image)
                else:
                    batch.append(cv_image)

            # TODO: Debugging
            end_time = self.clock.now()
            # self.get_logger().info(f"[{self.get_name()}] End processing {msg.header.frame_id} at {end_time.nanoseconds} nanoseconds")
            duration_ns = end_time.nanoseconds - self.start_time.nanoseconds
            duration_ms = duration_ns / 1e6  # Convert nanoseconds to seconds
            self.get_logger().info(f"undistort Function duration: {duration_ms:.3f} ms")


            # predict
            self.get_logger().info(f"the length of batch is {len(batch)}")
            results = self.yolo.predict(
                # source=cv_image,
                batch,
                verbose=False,
                stream=True,
                conf=self.threshold,
                iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                half=self.half,
                max_det=self.max_det,
                augment=self.augment,
                agnostic_nms=self.agnostic_nms,
                retina_masks=self.retina_masks,
                device=self.device,
            )


            if results is None:
                self.get_logger().error("YOLO results is None. Check if the source is properly set.")
                return

            try:
                results_list = list(results)
                self.get_logger().info(f"Number of results: {len(results_list)}")
            except TypeError as e:
                self.get_logger().error(f"Error converting results to list: {e}")
                return

            end_time = self.clock.now()
            duration_ns = end_time.nanoseconds - self.start_time.nanoseconds
            duration_ms = duration_ns / 1e6  # Convert nanoseconds to seconds
            self.get_logger().info(f"got results Function duration: {duration_ms:.3f} ms")

            for result in range(len(batch)):
                
                results: Results = results_list[result].cuda()
                # self.get_logger().info(f"results are {results}")

                if results.boxes or results.obb:
                    hypothesis = self.parse_hypothesis(results)
                    boxes = self.parse_boxes(results)

                # create detection msgs
                detections_msg = DetectionArray()

                for i in range(len(results)):

                    aux_msg = Detection()

                    if results.boxes or results.obb and hypothesis and boxes:
                        aux_msg.class_id = hypothesis[i]["class_id"]
                        aux_msg.class_name = hypothesis[i]["class_name"]
                        aux_msg.score = hypothesis[i]["score"]

                        aux_msg.bbox = boxes[i]

                    detections_msg.detections.append(aux_msg)
                
                # publish detections
                detections_msg.header = msg_batch[result].header # set the header for the correct detection 
                # Publish to the correct publisher based on frame_id

                target_pub = camera_publishers.get(msg_batch[result].header.frame_id, None)

                # If we found a matching publisher, publish the detections_msg                
                if target_pub:
                    final_pub.append(target_pub)
                    final_det.append(detections_msg)                   
        
            end_time = self.clock.now()
            duration_ns = end_time.nanoseconds - self.start_time.nanoseconds
            duration_ms = duration_ns / 1e6  # Convert nanoseconds to seconds
            self.get_logger().info(f"create detecton array Function duration: {duration_ms:.3f} ms")


            # Publish not empty detections
            for i in range(len(final_pub)):
                final_pub[i].publish(final_det[i])

            # Publish empty detections_msg to all other publishers
            for frame_id, pub in camera_publishers.items():                    
                if pub not in final_pub:
                    detections_msg_empty = DetectionArray()
                    detections_msg_empty.header = msg_batch[0].header
                    detections_msg_empty.header.frame_id = frame_id
                    pub.publish(detections_msg_empty)
            
            

            del results
            del cv_image
            del final_det
            del final_pub


    def set_classes_cb(
        self, req: SetClasses.Request, res: SetClasses.Response
    ) -> SetClasses.Response:
        self.get_logger().info(f"Setting classes: {req.classes}")
        self.yolo.set_classes(req.classes)
        self.get_logger().info(f"New classes: {self.yolo.names}")
        return res


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
