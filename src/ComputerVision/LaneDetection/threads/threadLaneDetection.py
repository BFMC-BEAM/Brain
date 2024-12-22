import cv2
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (mainCamera, serialCamera)
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
        self.subscribe()
        self.subscribers = {}
        super(threadLaneDetection, self).__init__()

    def run(self):
        while self._running:
            image = self.subscribers[serialCamera].receive()
            processor = LaneDetectionProcessor(type="simulador")
            out = processor.process_image(image)
            cv2.imshow("out", out)
            

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["Images"] = subscriber
        
