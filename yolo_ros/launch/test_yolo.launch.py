# Launch euclidean cluster and ray ground classifier nodes.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from iac_launch import get_share_file, append_launch_argument_dict

def generate_launch_description():
    """Launch euclidean cluster and ray ground classifier nodes."""
    # euclidean cluster parameter file definition.

    # TODO: Add the postprocessing param and yolo v5 param into camera launch pattern.
    #use image2body_params_sim.yaml if you'r running on SVL simulator.
    # camera_processing_param_file = get_share_file(
    #     'camera_bbox_processing', 'params', 'image2body_params.yaml')
    #use image2body_params_sim.yaml if you'r running on SVL simulator.
    yolo_cam_param_file = get_share_file(
        'yolo_ros', 'params', 'yolov11_params.yaml')
    #use image2body_params_sim.yaml if you'r running on SVL simulator.
    # yolov5_cam2_param_file = get_share_file(
        # 'yolov5_ros', 'params', 'yolov5_config.yaml')


    launch_entity_list = []
    launch_arg_dict = {}
    append_launch_argument_dict(launch_entity_list, launch_arg_dict,
        {
            "use_sim_time": ("False", "Use simulation clock if True"),
            # 'cluster_param_file': (cluster_param_file, 'Path to config file for euclidean clustering'),
            #'ground_param_file': (ground_param_file, 'Path to config file for ground filter'),
            # 'camera_processing_param_file': (camera_processing_param_file, 'Post processing parameters for camera images before YOLO v5 detection'),
            'camera_yolo_param_file': (yolo_cam_param_file, 'YOLO v11 parameter file for camera'),
            # 'camera_2_yolov5_param_file': (yolov5_cam2_param_file, 'YOLO v5 parameter file for camera 2'),
        }
    )

    # Camera Nodes:

    # camera_processing_node = Node(
    #     package='camera_bbox_processing',
    #     executable='image2body_frame_node',
    #     output='screen',
    #     parameters=[
    #         launch_arg_dict['camera_processing_param_file'],
    #     ],
    #     namespace='/perception/post',

    
    yolo_node = Node(   
        package='yolo_ros',
        executable='yolo_node',
        output='screen',
        parameters=[
           launch_arg_dict['camera_yolo_param_file'],
        ],

    )

    yolo_debug_node = Node(   
        package='yolo_ros',
        executable='debug_node',
        output='screen',
        parameters=[
           launch_arg_dict['camera_yolo_param_file'],
        ],

    )
    yolo_calc_bbox_node = Node(   
        package='yolo_ros',
        executable='calc_bbox_node',
        output='screen',
        parameters=[
           launch_arg_dict['camera_yolo_param_file'],
        ],

    )
    yolo_tracking_node = Node(   
        package='yolo_ros',
        executable='tracking_node',
        output='screen',
        parameters=[
           launch_arg_dict['camera_yolo_param_file'],
        ],

    )

    # yolov5_fcam1_node = Node(   
    #     package='yolov5_ros',
    #     executable='yolov5_fcam1',
    #     output='screen',
    #     parameters=[
    #         launch_arg_dict['camera_1_yolov5_param_file'],
    #     ]
    # )

    # yolov5_fcam2_node = Node(
    #     package='yolov5_ros',
    #     executable='yolov5_fcam2',
    #     output='screen',
    #     parameters=[
    #         launch_arg_dict['camera_2_yolov5_param_file'],
    #     ]
    # )

    launch_entity_list.extend([
        # Camera nodes:
        # camera_processing_node,
        yolo_node,
        yolo_calc_bbox_node,
        yolo_debug_node,
       
        # yolo_tracking_node,
        #yolov5_fcam1_node,
        # yolov5_fcam2_node
    ])

    return LaunchDescription(launch_entity_list)
