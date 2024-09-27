import cv2
import numpy as np
from typing import List, Tuple
import signal
import sys

class ObjectDetector:
    """
    A class for real-time object detection using YOLO.
    """

    def __init__(self, weights_path: str, config_path: str, classes_path: str):
        """
        Initialize the ObjectDetector with model paths.

        :param weights_path: Path to the YOLO weights file
        :param config_path: Path to the YOLO configuration file
        :param classes_path: Path to the file containing class names
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect objects in a single frame.

        :param frame: Input frame as a numpy array
        :return: List of tuples containing (class_name, confidence, bounding_box)
        """
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                if np.any(scores):
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                        w, h = int(detection[2] * width), int(detection[3] * height)
                        x, y = int(center_x - w/2), int(center_y - h/2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        results = []
        for i in indices:
            box = boxes[i]
            class_name = self.classes[class_ids[i]]
            confidence = confidences[i]
            results.append((class_name, confidence, tuple(box)))
        return results

    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        :param frame: Input frame
        :param detections: List of detections from detect_objects method
        :return: Frame with drawn detections
        """
        for class_name, confidence, (x, y, w, h) in detections:
            color = self.colors[self.classes.index(class_name)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

class ObjectDetectionApp:
    """
    A class to manage the object detection application.
    """

    def __init__(self, weights_path: str, config_path: str, classes_path: str):
        """
        Initialize the ObjectDetectionApp.

        :param weights_path: Path to the YOLO weights file
        :param config_path: Path to the YOLO configuration file
        :param classes_path: Path to the file containing class names
        """
        self.detector = ObjectDetector(weights_path, config_path, classes_path)
        self.cap = cv2.VideoCapture(0)
        self.is_running = False

    def signal_handler(self, sig, frame):
        """
        Handle interruption signals.

        :param sig: Signal number
        :param frame: Current stack frame
        """
        print("\nReceived interruption signal. Stopping the application...")
        self.stop()

    def start(self):
        """
        Start the object detection application.
        """
        self.is_running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detector.detect_objects(frame)
            frame_with_detections = self.detector.draw_detections(frame, detections)

            cv2.imshow("Real-time Object Detection", frame_with_detections)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self):
        """
        Stop the object detection application and release resources.
        """
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application stopped. Resources released.")
        sys.exit(0)

def main():
    """
    Main function to run the real-time object detection application.
    """
    # Paths to the pre-trained model and configuration files
    # We'll use the YOLO (You Only Look Once) model, which is popular for 
    # real-time object detection.
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"

    # List of class labels that can be detected
    classes_path = "coco.names"

    app = ObjectDetectionApp(weights_path, config_path, classes_path)
    app.start()

if __name__ == "__main__":
    main()