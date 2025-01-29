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
        self.declare_parameter("image_input_topic", "/perception/camera_front_center/image")
        self.declare_parameter("detection_topic", "/perception/post/camera_front_center/detections")

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
        self.image_input_topic = self.get_parameter("image_input_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value

        # set image qos
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        # detection publisher
        self._pub = self.create_lifecycle_publisher(DetectionArray, self.detection_topic, 10)
        
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

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

        self._sub = self.create_subscription(
            Image, self.image_input_topic, self.image_cb, self.image_qos_profile
        )

        # self.cam_info_sub = message_filters.Subscriber(
        #     self, CameraInfo, "/perception/camera_front_center/camera_info", 10
        # )

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

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)

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

    def image_cb(self, msg: Image) -> None:

        if self.enable:
            
            # convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # get camera info
            # cam_info = cam_info_ # TODO: based on the way the cam_info is published, re-write this 
            # Camera intrinsic parameters
            fx = 1007.495139
            fy = 1010.049539
            cx = 1067.657077
            cy = 746.317718
            mtx = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])

            # Distortion coefficients
            k1 = -0.167576
            k2 = 0.047559
            p1 = -0.000515
            p2 = 0.003160
            k3 = 0.0
            dist = np.array([k1, k2, p1, p2, k3])

            # Get image size
            h, w = cv_image.shape[:2]  # Assuming cv_image is the input image

            # Get the optimal new camera matrix
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
            # undistort the image before running YOLO
            # undistort
            dst = cv2.undistort(cv_image, mtx, dist, None, new_camera_mtx)

            # crop the image
            x, y, w, h = roi 
            dst = dst[y:y+h,x:x+w]

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
            self._pub.publish(detections_msg)

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
