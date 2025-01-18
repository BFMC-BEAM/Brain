import cv2
import torch

class ObjectDetectionProcessor:

    def __init__(self):
        # Cargar el modelo YOLOv5 nano
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
        # Dimensiones reales promedio de una señal de stop (en metros)
        self.real_width = 0.06  # señal de stop: 6 cm de ancho real
        self.focal_length = 309  # Focal length pixel raspberry camera v3

        

    def calculate_distance(self, bbox_width):
        """
        Calcula la distancia a la señal de stop usando la relación entre el ancho real y el ancho en la imagen.

        Args:
            bbox_width (int): Ancho de la caja delimitadora en píxeles.

        Returns:
            float: Distancia a la señal de stop en metros.
        """
        distance_m = (self.real_width * self.focal_length) / bbox_width
        distance_cm = distance_m * 100  # Convertir a centímetros
        return distance_cm

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
        signal_detected = df["name"].iloc[0] if not df.empty else None
        df = df[(df["confidence"] > 0.5) & (signal_detected == "stop sign")]

        valid_distance = True

        # Dibujar las cajas de detección en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            bbox_width = bbox[2] - bbox[0]
            distance_cm = self.calculate_distance(bbox_width)
            if distance_cm < 10:
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

        return (cv_image, signal_detected, valid_distance)
