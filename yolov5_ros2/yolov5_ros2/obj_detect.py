# ------------------------------------------------------------------------------------------------------------
# =========================================== YOLOv5 ROS2 ====================================================
# ------------------------------------------------------------------------------------------------------------
import time
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import numpy as np
from cv_bridge import CvBridge
from utils.image_publisher import *

# ------------------------------------------------------------------------------------------------------------
# Importing required ROS2 modules
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory

from boundingboxes.msg import BoundingBox, BoundingBoxes


class ImageStreamSubscriber(Node):

    def __init__(self):
        super().__init__('yolov5_ros2_node')
        
        # location of package
        package_share_directory = get_package_share_directory('yolov5_ros2')
        weight_loc = list()
        for direc in package_share_directory.split("/"):
            if direc != 'install' and direc != 'src' and direc != 'build':
                weight_loc.append(direc)
            else:
                break
        weight_loc.append("src/yolov5_ros2/yolov5_ros2/weights/")
        weight_loc = "/".join(weight_loc)
        #print(weight_loc)
        
        # parameters
        self.declare_parameter('weights', 'yolov5s.pt')
        self.declare_parameter('subscribed_topic', '/image')
        self.declare_parameter('published_topic', '/yolov5_ros2/image')
        self.declare_parameter('img_size', 416)
        self.declare_parameter('device', '')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 1000)
        self.declare_parameter('classes', None)
        self.declare_parameter('hide_labels', False)
        self.declare_parameter('hide_conf', False)
        self.declare_parameter('augment', True)
        self.declare_parameter('agnostic_nms', True)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('half', False)
        
        self.weights =  str(weight_loc) + self.get_parameter('weights').value
        self.published_topic = self.get_parameter('published_topic').value
        self.subscribed_topic = self.get_parameter('subscribed_topic').value
        self.imgsz = self.get_parameter('img_size').value
        self.device = self.get_parameter('device').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.classes = self.get_parameter('classes').value
        self.hide_labels = self.get_parameter('hide_labels').value
        self.hide_conf = self.get_parameter('hide_conf').value
        self.augment = self.get_parameter('augment').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.half = self.get_parameter('half').value

        check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
        self.bridge = CvBridge()
        
        # loading model
        self.model_initialization()
        
        # initializing publish and subscribe nodes
        self.flag = ord('a')
        self.detection_img_pub = self.create_publisher(Image, self.published_topic, 10)
        self.bboxes_pub = self.create_publisher(BoundingBoxes,"yolov5_ros2/bounding_boxes", 10)
        
        self.subscription = self.create_subscription(Image, self.subscribed_topic, self.subscriber_callback, 10)
        self.subscription                                                           # prevent unused variable warning

    def subscriber_callback(self, msg):
        
        # storing input image msg header
        imgmsg_header = msg.header
        
        # converting image-ros-msg into 3-channel (bgr) image formate
        self.im0s = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Padded resize
        self.img = letterbox(self.im0s, self.imgsz, stride=self.stride)[0]
        
        # Convert
        self.img = self.img.transpose((2, 0, 1))[::-1]                              # HWC to CHW, BGR to RGB
        self.img = np.ascontiguousarray(self.img)
        
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.half() if self.half else self.img.float()               # uint8 to fp16/32
        self.img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)
        
        # Inference
        self.t1 = time_synchronized()
        self.pred = self.model(self.img, augment=self.augment)[0]

        # Apply NMS
        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        self.t2 = time_synchronized()

        # Apply Classifier
        if self.classify:
            self.pred = apply_classifier(self.pred, self.modelc, self.img, self.im0s)
        
        # BoundingBoxes msg
        bboxes = BoundingBoxes()
        
        # Process detections
        for i, det in enumerate(self.pred):                                         # detections per image
            s, im0 = '', self.im0s.copy()
            #frame = getattr(dataset, 'frame', 0)
            s += '%gx%g ' % self.img.shape[2:]                                      # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]                              # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "               # add to string
                    
                for *xyxy, conf, cls in reversed(det):
                    
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness-1)    
                    
                    # Single BoundingBox msg
                    single_bbox = BoundingBox()
                    single_bbox.xmin = int(xyxy[0].item())
                    single_bbox.ymin = int(xyxy[1].item())
                    single_bbox.xmax = int(xyxy[2].item())
                    single_bbox.ymax = int(xyxy[3].item())
                    single_bbox.probability = conf.item()
                    single_bbox.id = c
                    single_bbox.class_id = self.names[c]
                    
                    bboxes.bounding_boxes.append(single_bbox)
        
        # Publishing bounding boxes and image with bounding boxes attached
        # same time-stamp for image and bounding box published, to match input image and output boundingboxes frames
        timestamp = (self.get_clock().now()).to_msg()
        
        processed_imgmsg = self.bridge.cv2_to_imgmsg(np.array(im0), encoding="bgr8")
        processed_imgmsg.header = imgmsg_header                                     # assigning header of input image msg
        processed_imgmsg.header.stamp = timestamp
        
        bboxes.header = imgmsg_header                                               # assigning header of input image msg
        bboxes.header.stamp = timestamp
        
        self.detection_img_pub.publish(processed_imgmsg)
        self.bboxes_pub.publish(bboxes)
        
        
    @torch.no_grad()
    def model_initialization(self):
        
        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.half and self.device.type != 'cpu'                         # half precision only supported on CUDA
        print("device:",self.device)
        
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)           # load FP32 model
        self.stride = int(self.model.stride.max())                                  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)                      # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        print("------------------Names of Classes------------------",self.names)
        
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)                    # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        
        return None
        


def main(args=None):
    rclpy.init(args=args)
    
    image_node = ImageStreamSubscriber()
    rclpy.spin(image_node)
    
    image_node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
