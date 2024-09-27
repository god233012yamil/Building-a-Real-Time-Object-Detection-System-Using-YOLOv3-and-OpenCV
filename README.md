# Building a Real-Time Object Detection System Using YOLOv3 and OpenCV

This repository provides a Python implementation of a real-time object detection system using the YOLOv3 (You Only Look Once) model and OpenCV. The program captures video input from a camera or video feed, detects objects using the YOLOv3 deep learning model, and displays the results with bounding boxes in real-time.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [YOLOv3 Files](#yolov3-files)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Code Structure](#code-structure)
   - [ObjectDetector Class](#objectdetector-class)
   - [Signal Handling](#signal-handling)
8. [Example Output](#example-output)
9. [Contributing](#contributing)
10. [License](#license)

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

yolov3.weights: Pre-trained weights file.
Download link: https://pjreddie.com/media/files/yolov3.weights
yolov3.cfg: YOLOv3 configuration file.
coco.names: File containing the names of the classes.
You can download these files from the official YOLOv3 website.

Place these files in the same directory as the Python script.


