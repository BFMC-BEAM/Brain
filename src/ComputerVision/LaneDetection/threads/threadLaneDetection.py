import cv2
import base64
import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CVCamera, serialCamera, Deviation, Direction, Lines, Intersection)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.ComputerVision.LaneDetection.lane_detection_onnx import LaneDetectionProcessor
import time

from src.utils.helpers import decode_image, encode_image
AVG_FRAME_COUNT = 15
class threadLaneDetection(ThreadWithStop):
    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.subscribers = {}
        self.subscribe()
        self.image_sender = messageHandlerSender(self.queuesList, CVCamera)
        self.deviation = messageHandlerSender(self.queuesList, Deviation)
        self.direction = messageHandlerSender(self.queuesList, Direction)
        self.lines = messageHandlerSender(self.queuesList, Lines)
        self.processor = LaneDetectionProcessor()
        super(threadLaneDetection, self).__init__()
        
        self.frame_count = 0
        self.deviation_history = []  # Lista para almacenar los últimos valores de desviación
        self.direction_history = []  # Lista para almacenar los últimos valores de desviación

    def run(self):
        while self._running:
            FrameCamera = self.subscribers["serialCamera"].receive()
            if FrameCamera is None:
                continue
            
            FrameCamera = encode_image(FrameCamera)
            e2, e3, _ = self.processor.process_image(FrameCamera)
            self.image_sender.send(decode_image(FrameCamera))

            # Agregamos la desviación a la lista
            self.deviation_history.append(e2)
            self.direction_history.append(e3)
            
            # Mantenemos solo los últimos 5 valores
            if len(self.deviation_history) > AVG_FRAME_COUNT:
                self.deviation_history.pop(0)

            # Mantenemos solo los últimos 5 valores
            if len(self.direction_history) > AVG_FRAME_COUNT:
                self.direction_history.pop(0)

            # Calculamos el promedio de los valores disponibles (mínimo 1, máximo 5)
            avg_deviation = sum(self.deviation_history) / len(self.deviation_history)
            avg_direction = sum(self.direction_history) / len(self.direction_history)

            # Enviar la dirección siempre
            self.direction.send(float(avg_direction))

            # Enviar el promedio de desviación
            self.deviation.send(float(avg_deviation))

            self.frame_count += 1

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "fifo", True)
        self.subscribers["serialCamera"] = subscriber

        