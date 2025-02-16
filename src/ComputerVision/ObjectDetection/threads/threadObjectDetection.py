import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CVCamera, serialCamera, CV_ObjectDetection_Type)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.ComputerVision.ObjectDetection.object_detection import ObjectDetectionProcessor
import time

class threadObjectDetection(ThreadWithStop):
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
        self.signal_type_detected = messageHandlerSender(self.queuesList, CV_ObjectDetection_Type)
        self.processor = ObjectDetectionProcessor()
        super(threadObjectDetection, self).__init__()
        self.act_deviation = 0.
        self.act_lines = -1 # contador de lineas detectadas, 0 nada, 1 si detecto izq o der, 2 normal


    def run(self):
        while self._running:
            FrameCamera = self.subscribers["serialCamera"].receive()

            if FrameCamera is None:
                continue
            start_time = time.time()


            decoded_image_data = base64.b64decode(FrameCamera)
            
            nparr = np.frombuffer(decoded_image_data, np.uint8)
            FrameCamera = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            FrameCameraPro, distance = self.processor.process_image(FrameCamera)
            
            _, serialEncodedImg = cv2.imencode(".jpg", FrameCameraPro)

            serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8")
            self.image_sender.send(serialEncodedImageData)
            
            if not distance:
                self.ObjectDetection_Type.send("stop_signal")
                #self.curr_sign = "stop_signal"
            else:
                self.ObjectDetection_Type.send("no_signal")
                #self.curr_sign = "no_signal"

            end_time = time.time()
            print(f"Computo de imagen en: {end_time - start_time} seg")

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["serialCamera"] = subscriber
        
 