# Building a Real-Time Object Detection System Using YOLOv3 and OpenCV

This repository provides a Python implementation of a real-time object detection system using the YOLOv3 (You Only Look Once) model and OpenCV. The program captures video input from a camera or video feed, detects objects using the YOLOv3 deep learning model, and displays the results with bounding boxes in real-time.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [YOLOv3 Files](#yolov3-files)
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

1. **Imports:**
   
![image](https://github.com/user-attachments/assets/305fac9d-d614-458b-8be0-2c7c487f3726)

The script imports essential libraries like cv2 for computer vision tasks, numpy for numerical operations, and signal and sys for managing termination signals.

2. **Class Definition: ObjectDetector:**

![image](https://github.com/user-attachments/assets/3fed422e-5460-47b0-b96c-9883456da688)

A class-based approach is used for object detection to encapsulate all related functions and variables. This class handles the initialization and running of the YOLO detection model.

3. **Initialization (__init__ method):**

![image](https://github.com/user-attachments/assets/e2e51667-9501-40ff-814c-5e9e308f43b5)

The constructor initializes the object detector with paths to the model's weights (weights_path), configuration (config_path), and class names (classes_path). It loads the YOLO model using OpenCV’s DNN module (cv2.dnn.readNet).

4. **Setting up the YOLO Network:**

![image](https://github.com/user-attachments/assets/0726d5c4-7b11-4343-b9e9-3583a479c718)

This helper method retrieves the names of the output layers from the YOLO network, which are essential for running object detection.

5. **Detection Function (detect_objects method):**

 ![image](https://github.com/user-attachments/assets/8a00ee18-ed98-48af-9377-a46fcc447285)
  
This method accepts a frame from the video feed, processes it into a blob suitable for the YOLO model (cv2.dnn.blobFromImage), and then runs the forward pass through the network to get the detection results.

6. **Drawing Bounding Boxes (draw_bounding_boxes method):**

![image](https://github.com/user-attachments/assets/b95e6508-cc14-41be-bb5e-ff23f9e57c6b)

This method processes the YOLO model’s outputs, extracts bounding box coordinates and class IDs, and draws them on the frame using cv2.rectangle and cv2.putText.

7. **Main Loop:**

![image](https://github.com/user-attachments/assets/9458c20c-adf8-4d17-ae17-2d5f7e0efda2)

The run method opens a video feed (or camera stream), continuously reads frames, detects objects, and displays the results with bounding boxes in real-time.

8. **Signal Handling for Graceful Exit:**

![image](https://github.com/user-attachments/assets/57845695-db7f-4a0b-976e-d3ffb4ea330d)

The script includes a signal handler that allows the program to exit cleanly when Ctrl+C is pressed. It ensures that resources are freed properly.

## License
This project is licensed under the GNU General Public License. See the LICENSE file for more information.

