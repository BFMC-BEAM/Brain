import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    CVCamera,
    CV_ObjectDetection_Type,
    Intersection)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.ComputerVision.ObjectDetection.object_detection import ObjectDetectionProcessor
import time

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
        self.image_sender = messageHandlerSender(self.queuesList, CVCamera)
        self.signal_type_detected = messageHandlerSender(self.queuesList, CV_ObjectDetection_Type)
        self.processor = ObjectDetectionProcessor()
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

            decoded_image_data = base64.b64decode(FrameCamera)
            nparr = np.frombuffer(decoded_image_data, np.uint8)
            FrameCamera = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            FrameCameraPro, distance = self.processor.process_image(FrameCamera)

            _, serialEncodedImg = cv2.imencode(".jpg", FrameCameraPro)
            serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8")
            self.image_sender.send(serialEncodedImageData)
            print("hay frame de objeto")
            
            if not distance:
                self.signal_type_detected.send("stop_signal")
            else:
                self.signal_type_detected.send("no_signal")

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, Intersection, "lastOnly", True)
        self.subscribers["Intersection"] = subscriber
 