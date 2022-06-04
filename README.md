# YOLOv5 + ROS2 : YOu can Leverage On ROS2
State-of-the-art object detection technique, [YOLOv5](https://github.com/ultralytics/yolov5) is now compatible with ROS2

## 1. Installation
### 1.1 Prerequisities
- Install ROS2 from the official [ROS2 website](https://www.ros.org/)<br>
For Ubuntu Linux - Focal Fossa (20.04): [ROS Foxy installation guide](https://docs.ros.org/en/foxy/Installation.html)
- Strictly necessary YOLOv5 related libraries and packages:<br>
Run the following command in terminal for [yolov5_strict_requirements.txt](yolov5_ros/setup.py) file:
```
pip install -r yolov5_strict_requirements.txt
```
### 1.2 Setting-up workspace
Run the following commands in terminal:
```
# create directory
mkdir -p ~/yolo_ws/src
cd ~/yolo_ws/src

# clone repo
git clone https://github.com/moksh-401-511/YOLOv5-ROS2-YOu-can-Leverage-On-ROS2.git

# build workspace
cd ~/yolo_ws
colcon build
```
## 2. Running Object Detection
### 2.1 Video input
Here we are using webcam for testing purpose. You can also use any other video source (like Intel d435).
Open terminal shell and run the following command to initialize webcam node:
```
ros2 run image_tools cam2image
```
**Note:** the above command will start publishing image-stream with topic name '/image'

### 2.2 YOLOv5 node
Now we will start our yolov5_ros2 node to perform object detection on input image-stream
Open another shell in terminal and run the following commands:
```
# move to our workspace directory
cd ~/yolo_ws

# source ws
. install/setup.bash

# run launch file to initialize yolov5_ros2 node
# provide the input image-stream topic name as argument (in our case, its '/image')
# also provide the weight file that you want to use (default: yolov5l.pt)
ros2 launch yolov5_ros2 yolov5_ros2_node.launch.py sub_topic:='/image' weights:='yolov5s.pt'
```
