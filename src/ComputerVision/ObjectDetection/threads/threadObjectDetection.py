import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    serialCamera,
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
            FrameCamera = self.subscribers["serialCamera"].receive()
            if FrameCamera is None:
                continue
            # limit object detection to run at specific intervals
            current_time = time.time()
            if(current_time - self.start_time < self.limit_time):
                continue
            self.start_time = time.time()

            FrameCamera = decode_image(FrameCamera)

            FrameCameraPro, objects = self.processor.process_image(FrameCamera)

            #serialEncodedImageData = encode_image(FrameCameraPro)
            #self.image_sender.send(serialEncodedImageData)
            #TODO: checkear el envio de listas
            encoded_objects = self._encode(objects)
            self.signals_detected.send(encoded_objects)


        
    def _encode(self, detected_signs):
        # Creamos una lista de strings formateados
        encoded_str = []
        for sign in detected_signs:
            class_name, valid_distance, lateral_distance = sign
            # Convertimos cada tupla en un string con los valores separados por comas
            sign_str = f"{class_name},{valid_distance},{lateral_distance}"
            encoded_str.append(sign_str)
        
        # Unimos todos los elementos con "*"
        return "*".join(encoded_str)

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["serialCamera"] = subscriber
    def _decode(self, encoded_str):
        # Dividimos la cadena por "*"
        sign_strings = encoded_str.split("*")
        
        detected_signs = []
        for sign_str in sign_strings:
            # Dividimos cada elemento por coma
            parts = sign_str.split(",")
            class_name = parts[0]
            valid_distance = parts[1] == 'True' if parts[1] != 'False' else False
            lateral_distance = None if parts[2] == 'None' else float(parts[2])
            
            # Agregamos la tupla a la lista
            detected_signs.append((class_name, valid_distance, lateral_distance))
        
        return detected_signs