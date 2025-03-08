import cv2
import numpy as np
from ultralytics import YOLO
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Silencia los mensajes de YOLO
class ObjectDetectionProcessor:
    def __init__(self):
        self.model_path = "best.pt"
        self.model = YOLO(self.model_path, verbose=False)  
        self.class_names = self.model.names
        self.real_width = 0.06      # Real width of the stop sign (in meters, 6 cm in this case)
        self.focal_length = 309     # Focal length of the Raspberry Pi V3 camera (in pixels)
        self.conf_threshold = 0.4   # Minimum confidence threshold for valid detection

        # Handlers for traffic signals
        self.traffic_signals_handlers = {
            "Stop_sign": self.stop_handler,
            "Parking_Sign": self.parking_handler,
            "Pedestrian_Sign": self.crosswalk_handler,
            "Priority_sign": self.priority_handler,
            "Motorway_start_sign": self.highway_entry_handler,
            "Motorway_finish_sign": self.highway_end_handler,
            "One_Way_Sign": self.one_way_handler,
            "Roundabout_sign": self.roundabout_handler,
            "Deny_sign": self.no_entry_handler,
        }

        # Handlers for obstacles
        self.obstacle_handlers = {
            "Pedestrian": self.person_handler,
            "Car": self.car_handler,
            "white_car": self.car_handler,
        }

    def calculate_distance(self, bbox_width):
        if bbox_width == 0:
            return float('inf')
        distance_m = (self.real_width * self.focal_length) / bbox_width
        return distance_m * 100  # Convert meters to centimeters

    def calculate_lateral_distance(self, bbox, image_width, bbox_width, distance_cm):
        center_x = image_width / 2
        object_x = (bbox[0] + bbox[2]) / 2
        offset = object_x - center_x
        real_x = ((offset * self.real_width) / bbox_width) * (distance_cm / self.real_width)
        return real_x

    def process_image(self, cv_image):
        image_width = cv_image.shape[1]
        results = self.model(cv_image)
        detected_signs = []
        detected_obstacles = []

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

                if class_name in self.traffic_signals_handlers:
                    handler = self.traffic_signals_handlers[class_name]
                    valid_distance, _ = handler(cv_image, bbox, confidence, image_width)
                    detected_signs.append((class_name, valid_distance))

                elif class_name in self.obstacle_handlers:
                    handler = self.obstacle_handlers[class_name]
                    valid_distance, lateral_distance = handler(cv_image, bbox, confidence, image_width)
                    detected_obstacles.append((class_name, valid_distance, lateral_distance))
        
        return cv_image, detected_signs, detected_obstacles

    def stop_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "STOP", image_width)
    
    def parking_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "PARKING", image_width)
    
    def crosswalk_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "CROSSWALK", image_width)
    
    def priority_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "PRIORITY", image_width)
    
    def highway_entry_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "HIGHWAY ENTRY", image_width)
    
    def highway_end_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "HIGHWAY END", image_width)
    
    def one_way_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "ONE WAY", image_width)
    
    def roundabout_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "ROUNDABOUT", image_width)
    
    def no_entry_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "NO ENTRY", image_width)
    
    def person_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "PERSON", image_width)
    
    def car_handler(self, cv_image, bbox, confidence, image_width):
        return self.handle_sign(cv_image, bbox, confidence, 40, "CAR", image_width)

    def handle_sign(self, cv_image, bbox, confidence, distance_threshold, sign_name, image_width):
        bbox_width = bbox[2] - bbox[0]
        distance_cm = self.calculate_distance(bbox_width)
        lateral_distance = self.calculate_lateral_distance(bbox, image_width, bbox_width, distance_cm)
        bbox_color = (255, 0, 0) if distance_cm < distance_threshold else (0, 0, 255)
        valid_distance = distance_cm <= distance_threshold
        confidence = round(confidence, 2)
        text = f"{sign_name}: {confidence:.2f}"
        text += f"\nDist: {distance_cm:.2f}cm"
        text += f"\nLat: {lateral_distance:.2f}cm"
        self.draw_sign_values(cv_image, bbox, text, bbox_color)
        return valid_distance, lateral_distance

    def draw_sign_values(self, cv_image, bbox, text, color):
        cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        lines = text.split("\n")
        x, y = bbox[0], bbox[1] - 15  # Posición inicial del texto
        for line in lines:
            cv2.putText(cv_image, line, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            y -= 15  # Espacio entre líneas
