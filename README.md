# Building a Real-Time Object Detection System Using YOLOv3 and OpenCV

This repository provides a Python implementation of a real-time object detection system using the YOLOv3 (You Only Look Once) model and OpenCV. The program captures video input from a camera or video feed, detects objects using the YOLOv3 deep learning model, and displays the results with bounding boxes in real-time.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [YOLOv3 Files](#yolov3-files)
5. [Code Structure](#code-structure)
   - [ObjectDetector Class](#objectdetector-class)
   - [Signal Handling](#signal-handling)
5. [Step-by-Step Code Explanation](#step-by-step-code-explanation) 
6. [License](#license)

## Overview

This project is built to demonstrate the power of real-time object detection using YOLOv3 and OpenCV. YOLOv3 is a highly efficient deep learning-based algorithm for detecting multiple objects in real time with high accuracy. The script processes video streams (like webcam feeds), detects objects, and draws bounding boxes around them, labeling the detected objects.

The script is written using a class-based structure for modularity and easy extension, with functionality to gracefully handle termination signals.

## Features

- Real-time object detection with YOLOv3 using OpenCV.
- Detects multiple objects per frame.
- Processes video streams from a camera or video file.
- Displays bounding boxes and labels for detected objects.
- Graceful program exit using signal handling (Ctrl+C).

## Requirements

- **Python 3.x** 
- **OpenCV 4.x** 
- **Numpy** 
- **YOLOv3 model weights, configuration file, and class names file**

Install the required Python libraries using the following command:

pip install opencv-python numpy

## YOLOv3 Files
To use YOLOv3 for object detection, you need to download the following files:

1. yolov3.weights: Pre-trained weights file.
- Download link: https://pjreddie.com/media/files/yolov3.weights
2. yolov3.cfg: YOLOv3 configuration file.
- Download link: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
3. coco.names: File containing the names of the classes.
- Download link: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

Place these files in the same directory as the Python script.

## Code Structure
**ObjectDetector Class**

The ObjectDetector class encapsulates the core functionalities of the YOLOv3 object detection system. Here’s a breakdown of its main components:

- _init_ method: Initializes the YOLO model with the provided paths to weights, configuration, and class names.
- _get_output_layers method: Retrieves the names of the YOLO model’s output layers needed for detection.
- detect_objects method: Preprocesses each video frame, runs the YOLO model, and returns detection outputs.
- draw_bounding_boxes method: Draws bounding boxes and class labels on detected objects in the video frame.
- run method: Manages video capture, processes each frame in real-time, performs object detection, and displays the output.

**Signal Handling**

To ensure a clean exit from the program, the script handles the SIGINT signal (Ctrl+C) to stop the program and release resources properly. This is achieved using:

![image](https://github.com/user-attachments/assets/91b5a848-2f92-4aa3-a6d1-00c862124812)

## Step-by-Step Code Explanation

1. Imports:
   ![image](https://github.com/user-attachments/assets/305fac9d-d614-458b-8be0-2c7c487f3726)
- The script imports essential libraries like cv2 for computer vision tasks, numpy for numerical operations, and signal and sys for managing termination signals.


## License
This project is licensed under the GNU General Public License. See the LICENSE file for more information.

