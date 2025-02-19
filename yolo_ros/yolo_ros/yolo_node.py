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

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
from ultralytics import YOLO, NAS, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
import message_filters


from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
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
        self.declare_parameter("model", "yolo_v11s.pt")
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

        self.fc_detection_topic = self.get_parameter("fc_detection_topic").value
        self.rc_detection_topic = self.get_parameter("rc_detection_topic").value
        self.rs_detection_topic = self.get_parameter("rs_detection_topic").value
        self.ls_detection_topic = self.get_parameter("ls_detection_topic").value
        self.fr_detection_topic = self.get_parameter("fr_detection_topic").value
        self.fl_detection_topic = self.get_parameter("fl_detection_topic").value

        self.undistort = self.get_parameter("undistort").value

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

        # set boolean flags
        self.cam_info_done = 0

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

        # camera topic subscribers
        self.fc_sub = self.create_subscription(
            Image, self.fc_input_topic, self.image_cb, self.image_qos_profile
        )
        self.rc_sub = self.create_subscription(
            Image, self.rc_input_topic, self.image_cb, self.image_qos_profile
        )
        self.rs_sub = self.create_subscription(
            Image, self.rs_input_topic, self.image_cb, self.image_qos_profile
        )
        self.ls_sub = self.create_subscription(
            Image, self.ls_input_topic, self.image_cb, self.image_qos_profile
        )
        self.fr_sub = self.create_subscription(
            Image, self.fr_input_topic, self.image_cb, self.image_qos_profile
        )
        self.fl_sub = self.create_subscription(
            Image, self.fl_input_topic, self.image_cb, self.image_qos_profile
        )
       
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


    def image_cb(self, msg: Image) -> None:

        if self.enable:
            
            # convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # if unndistort is enabled and cam info is stored, then undistort the image
            if self.undistort and self.cam_info_done == 6:

                # get camera info
                if msg.header.frame_id == "camera_front_center": 
                    k_mtx = self.fc_k_mtx
                    d_mtx = self.fc_d_mtx
                if msg.header.frame_id == "camera_rear_center": 
                    k_mtx = self.rc_k_mtx
                    d_mtx = self.rc_d_mtx
                if msg.header.frame_id == "camera_right_side": 
                    k_mtx = self.rs_k_mtx
                    d_mtx = self.rs_d_mtx
                if msg.header.frame_id == "camera_left_side": 
                    k_mtx = self.ls_k_mtx
                    d_mtx = self.ls_d_mtx
                if msg.header.frame_id == "camera_front_right": 
                    k_mtx = self.fr_k_mtx
                    d_mtx = self.fr_d_mtx
                if msg.header.frame_id == "camera_front_left": 
                    k_mtx = self.fl_k_mtx
                    d_mtx = self.fl_d_mtx
        
                # Get image size
                h, w = cv_image.shape[:2]  # Assuming cv_image is the input image

                # Get the optimal new camera matrix
                new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(k_mtx, d_mtx, (w, h), 0, (w, h))

                # undistort the image before running YOLO
                dst = cv2.undistort(cv_image, k_mtx, d_mtx, None, new_camera_mtx)

                # crop the image
                x, y, w, h = roi 
                un_dist_image = dst[y:y+h,x:x+w]

                # predict
                results = self.yolo.predict(
                    source=un_dist_image,
                    verbose=False,
                    stream=False,
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

            # if undistort is not enabled or cam info is not yet stored
            else:
                # predict
                results = self.yolo.predict(
                    source=cv_image,
                    verbose=False,
                    stream=False,
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

            results: Results = results[0].cuda()

            if results.boxes or results.obb:
                hypothesis = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results)

            if results.masks:
                masks = self.parse_masks(results)

            if results.keypoints:
                keypoints = self.parse_keypoints(results)

            # create detection msgs
            detections_msg = DetectionArray()

            for i in range(len(results)):

                aux_msg = Detection()

                if results.boxes or results.obb and hypothesis and boxes:
                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]

                    aux_msg.bbox = boxes[i]

                if results.masks and masks:
                    aux_msg.mask = masks[i]

                if results.keypoints and keypoints:
                    aux_msg.keypoints = keypoints[i]

                detections_msg.detections.append(aux_msg)

            # publish detections
            detections_msg.header = msg.header
            if msg.header.frame_id == "camera_front_center":
                self.fc_pub.publish(detections_msg)
            if msg.header.frame_id == "camera_rear_center":
                self.rc_pub.publish(detections_msg)
            if msg.header.frame_id == "camera_right_side":
                self.rs_pub.publish(detections_msg)
            if msg.header.frame_id == "camera_left_side":
                self.ls_pub.publish(detections_msg)
            if msg.header.frame_id == "camera_front_right":
                self.fr_pub.publish(detections_msg)
            if msg.header.frame_id == "camera_front_left":
                self.fl_pub.publish(detections_msg)

            del results
            del cv_image


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
