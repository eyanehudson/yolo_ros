import cv2
import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import math
import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import KeyPoint3DArray
from yolo_msgs.msg import BoundingBox3D


class BboxNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("calc_bbox_node")

        # parameters
        self.declare_parameter("target_frame", "center_of_gravity")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter(
            "cam_image_reliability", QoSReliabilityPolicy.RELIABLE
        )
        self.declare_parameter("cam_info_reliability", QoSReliabilityPolicy.RELIABLE)
        self.declare_parameter("dallara_height", 0.75)
        self.declare_parameter("scale", 0.5)


        self.declare_parameter("fc_detection_topic", "/perception/post/camera_front_center/detections")
        self.declare_parameter("rc_detection_topic", "/perception/post/camera_rear_center/detections")
        self.declare_parameter("rs_detection_topic", "/perception/post/camera_right_side/detections")
        self.declare_parameter("ls_detection_topic", "/perception/post/camera_left_side/detections")
        self.declare_parameter("fr_detection_topic", "/perception/post/camera_front_right/detections")
        self.declare_parameter("fl_detection_topic", "/perception/post/camera_front_left/detections")

        # aux
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.maximum_detection_threshold = (
            self.get_parameter("maximum_detection_threshold")
            .get_parameter_value()
            .double_value
        )
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
        cam_image_reliability = (
            self.get_parameter("cam_image_reliability")
            .get_parameter_value()
            .integer_value
        )
        self.dallara_height = self.get_parameter("dallara_height").value
        self.scale = self.get_parameter("scale").value

        self.fc_detection_topic = self.get_parameter("fc_detection_topic").value
        self.rc_detection_topic = self.get_parameter("rc_detection_topic").value
        self.rs_detection_topic = self.get_parameter("rs_detection_topic").value
        self.ls_detection_topic = self.get_parameter("ls_detection_topic").value
        self.fr_detection_topic = self.get_parameter("fr_detection_topic").value
        self.fl_detection_topic = self.get_parameter("fl_detection_topic").value

        self.cam_image_qos_profile = QoSProfile(
            reliability=cam_image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        cam_info_reliability = (
            self.get_parameter("cam_info_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.cam_info_qos_profile = QoSProfile(
            reliability=cam_info_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self.fc_pub = self.create_publisher(DetectionArray, "/perception/post/camera_front_center/detections_bbox", 10)
        self.rc_pub = self.create_publisher(DetectionArray, "/perception/post/camera_rear_center/detections_bbox", 10)
        self.rs_pub = self.create_publisher(DetectionArray, "/perception/post/camera_right_side/detections_bbox", 10)
        self.ls_pub = self.create_publisher(DetectionArray, "/perception/post/camera_left_side/detections_bbox", 10)
        self.stereo_pub = self.create_publisher(DetectionArray, "/perception/post/camera_front_stereo/detections_bbox", 10)


        # flags
        self.cam_info_done = 0

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


        # detection subs 
        self.fc_detections_sub = message_filters.Subscriber(
            self, DetectionArray, self.fc_detection_topic, qos_profile=10
        )
        self.rc_detections_sub = message_filters.Subscriber(
            self, DetectionArray, self.rc_detection_topic, qos_profile=10
        )
        self.rs_detections_sub = message_filters.Subscriber(
            self, DetectionArray, self.rs_detection_topic, qos_profile=10
        )
        self.ls_detections_sub = message_filters.Subscriber(
            self, DetectionArray, self.ls_detection_topic, qos_profile=10
        )
        self.fr_detections_sub = message_filters.Subscriber(
            self, DetectionArray, self.fr_detection_topic, qos_profile=10
        )
        self.fl_detections_sub = message_filters.Subscriber(
            self, DetectionArray, self.fl_detection_topic, qos_profile=10
        )

        # Detections should always be published as an empty list, so these should always synchronize. If the function is not entering at all, then check to ensure the detections are being published

        self.stereo_synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.fr_detections_sub, self.fl_detections_sub), 10, 0.5 
        )
        self.stereo_synchronizer.registerCallback(self.on_stereo_detections)

        self.pinhole_synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.fc_detections_sub, self.rc_detections_sub, self.rs_detections_sub, self.ls_detections_sub), 10, 0.5 
        )
        self.pinhole_synchronizer.registerCallback(self.on_pinhole_detections)


        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.cam_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener

        self.destroy_publisher(self._pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

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

    def on_stereo_detections(
        self,
        fr_detections_msg: DetectionArray,
        fl_detections_msg: DetectionArray,
    ) -> None:
        
        # self.get_logger().info(f"[{self.get_name()}] started on_stereo publisher")
        new_detections_msg = DetectionArray()
        new_detections_msg.header = fr_detections_msg.header
        new_detections_msg.header.frame_id = "front_stereo"
        new_detections_msg.detections = self.process_stereo_detections(
            fr_detections_msg, fl_detections_msg
        )
        self.stereo_pub.publish(new_detections_msg)

    def on_pinhole_detections(
        self,
        fc_detections_msg: DetectionArray,
        rc_detections_msg: DetectionArray,
        rs_detections_msg: DetectionArray,
        ls_detections_msg: DetectionArray,
    ) -> None:
        # self.get_logger().info(f"[{self.get_name()}] Entering on pinhole detection")

        fc_new_detections_msg = DetectionArray()
        rc_new_detections_msg = DetectionArray()
        rs_new_detections_msg = DetectionArray()
        ls_new_detections_msg = DetectionArray()

        fc_new_detections_msg.header = fc_detections_msg.header
        rc_new_detections_msg.header = fc_detections_msg.header
        rs_new_detections_msg.header = fc_detections_msg.header
        ls_new_detections_msg.header = fc_detections_msg.header


        fc_new_detections_msg.detections, rc_new_detections_msg.detections, rs_new_detections_msg.detections, ls_new_detections_msg.detections = self.process_pinhole_detections(
            fc_detections_msg, rc_detections_msg, rs_detections_msg, ls_detections_msg
        )
        
        self.fc_pub.publish(fc_new_detections_msg)
        self.rc_pub.publish(rc_new_detections_msg)
        self.rs_pub.publish(rs_new_detections_msg)
        self.ls_pub.publish(ls_new_detections_msg)
        # self.get_logger().info(f"[{self.get_name()}] new_bbox published")

    def process_stereo_detections(
        self,
        fr_detections_msg: DetectionArray,
        fl_detections_msg: DetectionArray,
    ) -> List[Detection]:

        # check if there are detections. must have detections in both cameras
        if (
            not fr_detections_msg.detections
            or not fl_detections_msg.detections
        ):
            # self.get_logger().info("not getting detections")
            return []

        # transform = self.get_transform(image_msg.header.frame_id) # This might need to be commented out for now

        # if transform is None:
        #     return []

        new_detections = []
            
        for fr_detection, fl_detection in zip(fr_detections_msg.detections, fl_detections_msg.detections):
            bbox3d = self.convert_stereo(fr_detection, fl_detection)

            if bbox3d is not None:
                fr_detection.bbox3d = bbox3d         #TODO: Look at this in depth. Should be saving detections info from fr, but not sure how to handle this        
                # bbox3d = BboxNode.transform_3d_box(bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
                new_detections.append(fr_detection)

        return new_detections

    def process_pinhole_detections(
        self,
        fc_detections_msg: DetectionArray,
        rc_detections_msg: DetectionArray,
        rs_detections_msg: DetectionArray,
        ls_detections_msg: DetectionArray,
    ) -> Tuple[List[Detection], List[Detection], List[Detection], List[Detection]]:

        # self.get_logger().info(f"{[self.get_name()]} Entering pinhole detection")
        # check if there are detections
        if (
            not fc_detections_msg.detections
            and not rc_detections_msg.detections
            and not rs_detections_msg.detections
            and not ls_detections_msg.detections
        ):
            return [], [], [], []
        

        # transform = self.get_transform(image_msg.header.frame_id) # This might need to be commented out for now

        # if transform is None:
        #     return []

        # List of camera messages with names and their corresponding result lists
        camera_detections = [
            ("front_center", fc_detections_msg, []),  # List for fc_new_detections
            ("rear_center", rc_detections_msg, []),   # List for rc_new_detections
            ("right_side", rs_detections_msg, []),    # List for rs_new_detections
            ("left_side", ls_detections_msg, []),     # List for ls_new_detections
        ]

        for cam_name, detections_msg, new_detections in camera_detections:
            if not detections_msg.detections:
                continue  # Skip empty detections

            for detection in detections_msg.detections:
                bbox3d = self.convert_pinhole(detection, cam_name)
                if bbox3d is not None:
                    detection.bbox3d = bbox3d
                    bbox3d.frame_id = self.target_frame
                    new_detections.append(detection)

        # Extract individual lists to return
        fc_new_detections = camera_detections[0][2]
        rc_new_detections = camera_detections[1][2]
        rs_new_detections = camera_detections[2][2]
        ls_new_detections = camera_detections[3][2]

        return fc_new_detections, rc_new_detections, rs_new_detections, ls_new_detections

    def convert_pinhole(
        self, 
        detection: Detection, 
        cam_name: str
        
    ) -> BoundingBox3D:

        center_x = detection.bbox.center.position.x # in pixels
        self.get_logger().info(f"center_x is = {center_x}")

        # center_y = int(detection.bbox.center.position.y) # in pixels
        # size_x = int(detection.bbox.size.x) # in pixels width of car
        size_y = detection.bbox.size.y # in pixels height of car

        # Get k matrix for correct cam
        if cam_name == "front_center":
            k_mtx = self.fc_k_mtx
        elif cam_name == "rear_center":
            k_mtx = self.rc_k_mtx
        elif cam_name == "right_side":
            k_mtx = self.rs_k_mtx
        elif cam_name == "left_side":
            k_mtx = self.ls_k_mtx  
        fy = k_mtx[1,1]
        fx = k_mtx[0,0]
        cx = k_mtx[0,2] * self.scale # TODO: Need to scale the centroid based on downsampling of the image in processing 
        cy = k_mtx[1,2] * self.scale

        # TODO: Correct the pinhole model calculations
        depth = (fy * self.dallara_height)/(size_y) # forward is positive (in meters)
        self.get_logger().info(f"depth is = {depth}")
        self.get_logger().info(f"cx =  is = {cx}")

        real_y = - (center_x - cx)/(fx * depth) # left is positive (in meters) # TODO: def wrong algo
        self.get_logger().info(f"y  is = {real_y}")


        
        # Compute the angle to the car based on x and y
        if depth == 0: # this should never happen
            theta = 0.0
        else:
            theta = math.atan(real_y / depth)  # Theta in radians
            theta_degrees = math.degrees(theta)


        # create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = depth
        msg.center.position.y = real_y
        msg.center.position.z = theta_degrees # publishing as theta for now

        return msg

    def convert_stereo(
            self,
            fr_detections_msg: Detection,
            fl_detections_msg: Detection,
        ) -> BoundingBox3D:
        # TODO: potentially replace with M matrix least square estimate
        cam_distance = .36 # TODO: in meters. math based on urdf24. need to check and maybe calculate using least squares
        fx = self.fr_k_mtx[0,0] # TODO: crude estimate based only on fr cam, either average the values or calculate using least squares
        fr_cx = self.fr_k_mtx[0,2]
        fl_cx = self.fl_k_mtx[0,2]
        # self.get_logger().info(f"fl_cx = {fl_cx},    fr_cx = {fr_cx}")
        # do something
        # calculate the disparity

        # 1. rectify the two stereo cameras
        # 2. make the assumption that the car detected is on the same y. only x changes
        # 3. calculate the disparity "X", where x is the left camera and x_prime is the right camera
        # 4. try taking cx - x where cx is the centroid x position and x is the centroid position of the bbox
        x = fl_detections_msg.bbox.center.position.x - fl_cx # left disparity. Should be negative if car is on the right of centerline
        x_prime = fr_detections_msg.bbox.center.position.x - fr_cx  # right disparity
        disparity = x - x_prime
        # self.get_logger().info(f"x = {x},    x_prime = {x_prime},     disparity = {disparity}")
        if disparity == 0: # handle the zero case
            disparity = 0.0000001 # this will make depth inf

        depth = cam_distance*fx/(disparity) #TODO: correct this math
        self.get_logger().info(f"depth = {depth}")
        lat_dist = - ((x+x_prime)/2 * depth)/fx

        # Compute the angle to the car based on x and y
        if depth == 0: # this should never happen
            theta = 0.0
        else:
            theta = math.atan(lat_dist / depth)  # Theta in radians
            theta_degrees = math.degrees(theta)

        # create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = depth
        msg.center.position.y = lat_dist
        msg.center.position.z = theta_degrees # publishing as theta for now
        
        return msg

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]: 
        # transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D, translation: np.ndarray, rotation: np.ndarray
    ) -> BoundingBox3D:

        # position
        position = (
            BboxNode.qv_mult(
                rotation,
                np.array(
                    [
                        bbox.center.position.x,
                        bbox.center.position.y,
                        bbox.center.position.z,
                    ]
                ),
            )
            + translation
        )

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # # size
        # size = Detect3DNode.qv_mult(
        #     rotation, np.array([bbox.size.x, bbox.size.y, bbox.size.z])
        # )

        # bbox.size.x = abs(size[0])
        # bbox.size.y = abs(size[1])
        # bbox.size.z = abs(size[2])

        return bbox

    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray, translation: np.ndarray, rotation: np.ndarray
    ) -> KeyPoint3DArray:

        for point in keypoints.data:
            position = (
                Detect3DNode.qv_mult(
                    rotation, np.array([point.point.x, point.point.y, point.point.z])
                )
                + translation
            )

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)


def main():
    rclpy.init()
    node = BboxNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
