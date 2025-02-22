import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CVCamera, serialCamera, Deviation, Direction, Lines, Intersection)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.ComputerVision.LaneDetection.lane_detection_onnx import LaneDetectionProcessor
import time

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
        self.deviation = messageHandlerSender(self.queuesList, Deviation)
        self.direction = messageHandlerSender(self.queuesList, Direction)
        self.intersection = messageHandlerSender(self.queuesList, Intersection)
        self.lines = messageHandlerSender(self.queuesList, Lines) #TODO: modificar nombre
        self.processor = LaneDetectionProcessor(type="simulator")
        super(threadLaneDetection, self).__init__()
        self.act_deviation = 0.
        self.act_lines = -1     # detected lines counter, 0 none, 1 if detected left or right, 2 normal


    def run(self):
        while self._running:
            FrameCamera = self.subscribers["serialCamera"].receive()
            if FrameCamera is None:
                continue
            start_time = time.time()

            decoded_image_data = base64.b64decode(FrameCamera)
            nparr = np.frombuffer(decoded_image_data, np.uint8)
            FrameCamera = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            e2, e3, _=self.processor.process_image(FrameCamera)
            
            #_, serialEncodedImg = cv2.imencode(".jpg", FrameCameraPro)
            #serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8")
            self.image_sender.send(FrameCamera)
            self.direction.send(e3)
            self.deviation.send(e2)
            self.act_deviation = e2
            #ret = self.processor.get_parameters(self.act_deviation)
            #new_cant_lines = self.processor.get_lines()
            
            #is_possible_signal = self.processor.get_in_possible_signal()
            #if new_cant_lines != self.act_lines:
            #    self.lines.send(new_cant_lines)
            #    self.act_lines = new_cant_lines
        

            #if is_possible_signal is True:
            #    self.intersection.send(serialEncodedImageData)

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["serialCamera"] = subscriber
        
        