import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetectionProcessor:
    def __init__(self):
        self.model_path = "best.pt"
        self.model = YOLO(self.model_path)  
        self.class_names = self.model.names
        self.real_width = 0.06      # Real width of the stop sign (in meters, 6 cm in this case)
        self.focal_length = 309     # Focal length of the Raspberry Pi V3 camera (in pixels)
        self.conf_threshold = 0.4   # Minimum confidence threshold for valid detection

        # Map class names to their corresponding handler functions
        self.signal_handlers = {
            "STOP": self.stop_handler,
            "PARKING": self.parking_handler,
            "CROSSWALK": self.crosswalk_handler,
            "PRIORITY": self.priority_handler,
            "HIGHWAY_ENTRY": self.highway_entry_handler,
            "HIGHWAY_END": self.highway_end_handler,
            "ONE_WAY": self.one_way_handler,
            "ROUNDABOUT": self.roundabout_handler,
            "NO_ENTRY": self.no_entry_handler,
            "PEDESTRIAN": self.pedestrian_handler,
            "CAR": self.car_handler,
        }

    def calculate_distance(self, bbox_width):
        if bbox_width == 0:
            return float('inf')
        distance_m = (self.real_width * self.focal_length) / bbox_width
        return distance_m * 100  # Convert meters to centimeters

    def process_image(self, cv_image):
        results = self.model(cv_image)
        detected_signs = []
        
        for result in results:
            boxes = result.boxes.xyxy.numpy().astype(int)
            confidences = result.boxes.conf.numpy().astype(float)
            labels = result.boxes.cls.numpy().astype(int)

            valid_indices = confidences >= self.conf_threshold
            boxes, confidences, labels = boxes[valid_indices], confidences[valid_indices], labels[valid_indices]

            for i in range(len(boxes)):
                bbox = boxes[i]
                confidence = confidences[i]
                label = labels[i]
                class_name = self.class_names[label]
                handler = self.signal_handlers.get(class_name, lambda *args: False)
                valid_distance = handler(cv_image, bbox, confidence)
                detected_signs.append((class_name, valid_distance))
        
        return cv_image, detected_signs

    def stop_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "STOP")
    
    def parking_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "PARKING")
    
    def crosswalk_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "CROSSWALK")
    
    def priority_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "PRIORITY")
    
    def highway_entry_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "HIGHWAY_ENTRY")
    
    def highway_end_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "HIGHWAY_END")
    
    def one_way_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "ONE_WAY")
    
    def roundabout_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "ROUNDABOUT")
    
    def no_entry_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "NO_ENTRY")
    
    def pedestrian_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "PEDESTRIAN")
    
    def car_handler(self, cv_image, bbox, confidence):
        distance_threshold = 40  # cm
        return self.handle_sign(cv_image, bbox, confidence, distance_threshold, "CAR")

    def handle_sign(self, cv_image, bbox, confidence, distance_threshold, sign_name):
        bbox_width = bbox[2] - bbox[0]
        distance_cm = self.calculate_distance(bbox_width)
        bbox_color = (255, 0, 0) if distance_cm < distance_threshold else (0, 0, 255)
        valid_distance = distance_cm >= distance_threshold
        confidence = round(confidence, 2)
        self.draw_sign_values(cv_image, bbox, f"{sign_name}: {confidence:.2f}, Dist: {distance_cm:.2f}cm", bbox_color)
        return valid_distance

    def draw_sign_values(self, cv_image, bbox, text, color):
        cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(cv_image, text, (bbox[0], bbox[1] - 15), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)