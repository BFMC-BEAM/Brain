import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    CVCamera,
    CV_ObjectsDetected,
    Intersection)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.ComputerVision.ObjectDetection.object_detection import ObjectDetectionProcessor
import time

from src.utils.helpers import decode_image, encode_image

class threadObjectDetection(ThreadWithStop):
    """This thread handles ObjectDetection.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False, yolo_model=None):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.subscribers = {}
        self.subscribe()
        self.image_sender = messageHandlerSender(self.queuesList, CVCamera)
        self.signals_detected = messageHandlerSender(self.queuesList, CV_ObjectsDetected)
        self.processor = ObjectDetectionProcessor(yolo_model)
        super(threadObjectDetection, self).__init__()
        self.start_time = time.time()
        self.limit_time = 3
        self.init_count_time = True

        


    def run(self):
        while self._running:
            # reads the queue that only sends information when the lineDetector detects an intersection
            FrameCamera = self.subscribers["Intersection"].receive()
            if FrameCamera is None:
                continue

            # limit object detection to run at specific intervals
            current_time = time.time()
            if(current_time - self.start_time < self.limit_time):
                continue
            self.start_time = time.time()

            FrameCamera = decode_image(FrameCamera)

            FrameCameraPro, signals = self.processor.process_image(FrameCamera)

            serialEncodedImageData = encode_image(FrameCameraPro)
            self.image_sender.send(serialEncodedImageData)
            #TODO: checkear el envio de listas
            self.signals_detected.send(signals)
        

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, Intersection, "lastOnly", True)
        self.subscribers["Intersection"] = subscriber
 