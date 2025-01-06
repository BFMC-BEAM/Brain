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
        self.processor = LaneDetectionProcessor(type="simulador")
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

                # Decodificar el array de numpy a una imagen OpenCV
                cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                out = self.processor.process_image(cv_image)
                serialEncodedImageData = base64.b64encode(out).decode("utf-8")

                self.image_sender.send(serialEncodedImageData)

            

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["Images"] = subscriber
        
