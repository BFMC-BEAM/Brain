import cv2
#import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetectionProcessor:

    def __init__(self):
        model = YOLO("yolov5s.pt")
        model.export(format="ncnn")  # Generate 'yolov5su_ncnn_model'

        self.model = YOLO("yolov5su_ncnn_model")
        self.class_names = self.model.names

        self.real_width = 0.06      # Señal de stop: 6 cm de ancho real
        self.focal_length = 309     # Focal length pixel raspberry camera v3



    def calculate_distance(self, bbox_width):
        """
        Calcula la distancia a la señal de stop usando la relación entre el ancho real y el ancho en la imagen.
        """
        if bbox_width == 0:
            return float('inf')
        
        distance_m = (self.real_width * self.focal_length) / bbox_width
        return distance_m * 100  # Convertir a cm



    def process_image(self, cv_image):
        """
        Procesa la imagen detectando objetos y midiendo la distancia a una señal de stop.
        """
        # Inferencia
        results = self.model(cv_image)

        # Obtener las detecciones
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

                bbox_color = (255, 0, 0) if distance_cm < 40 else (0, 0, 255)   #bbox azul si la distancia es menor a 40cm, rojo si es mayor
                valid_distance = distance_cm >= 40

                # Dibujar bounding box
                cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)

                # Dibujar texto con nombre, confianza y distancia
                cv2.putText(cv_image,
                            f"{class_name}: {round(confidences[i], 4)}, Dist: {round(distance_cm, 2)}cm",
                            (bbox[0], bbox[1] - 15),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 255, 255),
                            2)

        return (cv_image, valid_distance)

"""
class ObjectDetectionProcessor:

    def __init__(self):
        # Cargar el modelo YOLOv5 nano
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        # Dimensiones reales promedio de una señal de stop (en metros)
        self.real_width = 0.06  # señal de stop: 6 cm de ancho real
        self.focal_length = 309  # Focal length pixel raspberry camera v3

        

    def calculate_distance(self, bbox_width):
        """"""
        Calcula la distancia a la señal de stop usando la relación entre el ancho real y el ancho en la imagen.

        Args:
            bbox_width (int): Ancho de la caja delimitadora en píxeles.

        Returns:
            float: Distancia a la señal de stop en metros.
        """"""
        distance_m = (self.real_width * self.focal_length) / bbox_width
        distance_cm = distance_m * 100  # Convertir a centímetros
        return distance_cm

    def process_image(self, cv_image):
        """"""
        Processes the input image to detect a series of classes in our dataset.

        Args:
            cv_image (np.ndarray): Input image.

        Returns:
            np.ndarray: Processed output image with bounding boxes for any class in our dataset.
        """"""
        # Inferencia con YOLOv5
        pred = self.model(cv_image)
        
        # Convertir predicciones a un DataFrame y filtrar por confianza
        df = pred.pandas().xyxy[0]
        signal_detected = df["name"].iloc[0] if not df.empty else None
        
        if signal_detected != "stop sign":
            return (cv_image, True)

        df = df[(df["confidence"] > 0.7) & (signal_detected == "stop sign")]

        valid_distance = True

        # Dibujar las cajas de detección en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            bbox_width = bbox[2] - bbox[0]
            distance_cm = self.calculate_distance(bbox_width)
            if distance_cm < 40:
                bbox_color = (255, 0, 0)  # Azul
                valid_distance = False
            else:
                bbox_color = (0, 0, 255)  # Rojo

            cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)
            cv2.putText(cv_image,
                        f"Stop Sign: {round(df.iloc[i]['confidence'], 4)}, Dist: {round(distance_cm, 2)}cm",
                        (bbox[0], bbox[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        2)

        return (cv_image, valid_distance)"""