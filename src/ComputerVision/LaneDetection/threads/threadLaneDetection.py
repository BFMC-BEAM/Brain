import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CVCamera, serialCamera)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.ComputerVision.LaneDetection.lane_detection import LaneDetectionProcessor

class threadLaneDetection(ThreadWithStop):
    """This thread handles LaneDetection.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.subscribers = {}
        self.subscribe()
        self.image_sender = messageHandlerSender(self.queuesList, CVCamera)
        self.processor = LaneDetectionProcessor(type="simulator")
        super(threadLaneDetection, self).__init__()

    def run(self):
        while self._running:
            image = self.subscribers["Images"].receive()
            if image is not None:
                if image.startswith("data:image"):
                    image = image.split(",")[1]

                # Decodificar la imagen Base64 a bytes
                image_data = base64.b64decode(image)

                # Convertir los bytes a un array de numpy
                np_array = np.frombuffer(image_data, dtype=np.uint8)

                # Decodificar el array numpy a una imagen OpenCV
                cv_image = cv2.imdecode(np_array, cv2.COLOR_YUV2BGR_I420)  # Decodificar como imagen BGR
                print("antes", cv_image.shape[0], cv_image.shape[1])
                
                if cv_image is None:
                    print("Error: cv2.imdecode falló. Verifica el formato de la imagen.")
                    continue
                
                # Procesar la imagen decodificada
                out, deviation, direction = self.processor.process_image(cv_image)  # Captura los valores de deviation y direction

                # Volver a codificar el resultado si es necesario
                _, encoded_output = cv2.imencode(".jpg", out)
                serialEncodedImageData = base64.b64encode(encoded_output).decode("utf-8")
                self.image_sender.send(serialEncodedImageData)  # Enviar imagen procesada

            

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["Images"] = subscriber
        
