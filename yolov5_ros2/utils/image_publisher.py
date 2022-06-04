import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import numpy as np
from cv_bridge import CvBridge

published_topic = '/yolov5_detection/image'

class ImageStreamPublisher(Node):

    def __init__(self):
        super().__init__('image_stream_publisher')
        self.publisher_ = self.create_publisher(Image, published_topic, 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.publisher_callback)
        # dummy image for initializing variable
        self.image = np.zeros(240*320)

    def publisher_callback(self):
        self.publisher_.publish(self.image)


def publish_image(image):
    pub.image = bridge.cv2_to_imgmsg(np.array(image), "bgr8")
    rclpy.spin_once(pub)

def create_node():
    global pub
    global bridge
    bridge = CvBridge()
    pub = ImageStreamPublisher()
