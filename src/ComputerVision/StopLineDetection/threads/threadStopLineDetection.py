import cv2
import base64
import numpy as np
from src.ComputerVision.StopLineDetection.stopline_detection import StopLineDetectionProcessor
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    CVCamera,
    CV_ObjectsDetected,
    Intersection,
    serialCamera)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import time

from src.utils.helpers import decode_image, encode_image

class threadStopLineDetection(ThreadWithStop):
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
        self.stopline_sender = messageHandlerSender(self.queuesList, Intersection)
        self.processor = StopLineDetectionProcessor()
        super(threadStopLineDetection, self).__init__()
        self.start_time = time.time()
        self.limit_time = 3
        self.init_count_time = True

        


    def run(self):
        while self._running:
            # reads the queue that only sends information when the lineDetector detects an intersection
            FrameCamera = self.subscribers["serialCamera"].receive()
            if FrameCamera is None:
                continue

            # limit object detection to run at specific intervals
            current_time = time.time()
            if(current_time - self.start_time < self.limit_time):
                continue
            self.start_time = time.time()
            FrameCamera = decode_image(FrameCamera)
            dist, _,_ = self.processor.process_image(FrameCamera)
            serialEncodedImageData = encode_image(FrameCamera)
            if dist < 0.75:
                #self.stopline_sender.send(serialEncodedImageData)
                print(f"stopline detected at est {dist}")
        

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "fifo", True)
        self.subscribers["serialCamera"] = subscriber
 