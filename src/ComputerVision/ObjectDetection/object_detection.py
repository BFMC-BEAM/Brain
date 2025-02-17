import cv2
import numpy as np

class ObjectDetectionProcessor:

    def __init__(self, model):


        self.model = model
        self.class_names = self.model.names

        self.real_width = 0.06      # Stop signal: 6 cm actual width
        self.focal_length = 309     # Focal length pixel raspberry camera v3



    def calculate_distance(self, bbox_width):
        """
        Calculates the distance to the stop sign using the relationship between the actual width and the width in the image.
        """
        if bbox_width == 0:
            return float('inf')
        
        distance_m = (self.real_width * self.focal_length) / bbox_width
        return distance_m * 100  # Convertir a cm



    def process_image(self, cv_image):
        """
        Processes the image by detecting objects and measuring the distance to a stop sign.
        """
        # Inference
        results = self.model(cv_image)

        # Get detections and draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            labels = result.boxes.cls.numpy().astype(int)

            valid_distance = True  # Default

            for i, bbox in enumerate(boxes):
                if confidences[i] < 0.7:
                    continue

                class_name = self.class_names[labels[i]]

                if class_name != "stop sign":
                    continue

                bbox = bbox.astype(int)
                bbox_width = bbox[2] - bbox[0]
                distance_cm = self.calculate_distance(bbox_width)

                bbox_color = (255, 0, 0) if distance_cm < 40 else (0, 0, 255)   #blue bbox if the distance is less than 40cm, red if it is greater
                valid_distance = distance_cm >= 40

                # Draw bbox
                cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)

                # Draw text with name, confidence, and distance
                cv2.putText(cv_image,
                            f"{class_name}: {round(confidences[i], 4)}, Dist: {round(distance_cm, 2)}cm",
                            (bbox[0], bbox[1] - 15),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 255, 255),
                            2)

        return (cv_image, valid_distance)
