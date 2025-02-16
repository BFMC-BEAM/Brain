import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CVCamera, serialCamera)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.ComputerVision.ObjectDetection.object_detection import ObjectDetectionProcessor

class threadObjectDetection(ThreadWithStop):
    """This thread handles ObjectDetection.
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
        self.image_sender = messageHandlerSender(self.queuesList, serialCamera)
        self.processor = ObjectDetectionProcessor()
        super(threadObjectDetection, self).__init__()

    def run(self):
        while self._running:
            image = self.subscribers["Images"].receive()
            if image is not None:
                if image.startswith("data:image"):
                    image = image.split(",")[1]

                image_data = base64.b64decode(image)
                np_array = np.frombuffer(image_data, dtype=np.uint8)
                cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                
                if cv_image is None:
                    print("Error: cv2.imdecode failed.")
                    continue
                
                out = self.processor.process_image(cv_image)
                _, encoded_output = cv2.imencode(".jpg", out)
                serialEncodedImageData = base64.b64encode(encoded_output).decode("utf-8")
                #self.image_sender.send(serialEncodedImageData)

    def subscribe(self):
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["Images"] = subscriber