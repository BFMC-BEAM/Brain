import cv2
import numpy as np
import math
import torch

class ObjectDetectionProcessor:

    def __init__(self):
        # Cargar el modelo YOLOv5 nano
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)

    def process_image(self, cv_image):
        """
        Processes the input image to detect a series of classes in our dataset.

        Args:
            cv_image (np.ndarray): Input image.

        Returns:
            np.ndarray: Processed output image with bounding boxes for any class in our dataset.
        """
        # Inferencia con YOLOv5
        pred = self.model(cv_image)
        
        # Convertir predicciones a un DataFrame y filtrar por confianza
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] > 0.5]

        # Dibujar las cajas de detección en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(cv_image,
                        f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'], 4)}",
                        (bbox[0], bbox[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        2)

        return cv_image
