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
AVG_FRAME_COUNT = 3
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
            
            FrameCamera = decode_image(FrameCamera)
            e2, e3, _ = self.processor.process_image(FrameCamera)
           

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
            pa = np.array([SHOW_DIST*np.cos(e3),SHOW_DIST*np.sin(e3)])
            cv2.line(FrameCamera, project_onto_frame(pa, cam=FRONT_CAM), (FrameCamera.shape[1]//2, FrameCamera.shape[0]), YELLOW, 4) #pa frame 

            self.image_sender.send(encode_image(FrameCamera))


            self.frame_count += 1

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "fifo", True)
        self.subscribers["serialCamera"] = subscriber

BLACK = (0, 0, 0)
SHOW_DIST = 0.55 # distance ahead just to show
YELLOW = (0, 255, 255)
ZOFF = 0.03
CAM_PITCH = np.deg2rad(20)  # [rad]
CAM_FOV = 1.085594795  
CM2WB = 0.16                        # [m]       distance from center of mass to wheel base 0.22  
FRONT_CAM = {'fov':CAM_FOV,'θ':CAM_PITCH,'x':0.0+CM2WB,'z':0.2+ZOFF, 'w':320, 'h':240}
def project_onto_frame(points, cam=FRONT_CAM):
    ''' function to project points onto a camera frame, returns the points in pixel coordinates '''
    assert isinstance(points, np.ndarray), f'points must be np.ndarray, got {type(points)}'
    assert points.shape[-1] == 2, f'points must be (something,2), got {points.shape}'
    assert points.ndim <= 2, f'points must be (something,2), got {points.shape}'
    θ, xc, zc, w, h = cam['θ'], cam['x'], cam['z'], cam['w'], cam['h']
    pts = points.reshape(-1, 2) #flatten points
    pts = np.concatenate((pts, np.zeros((pts.shape[0],1))), axis=1) # and add z coordinate
    R, T = np.array([[np.cos(θ), 0, np.sin(θ)], [0, 1, 0], [-np.sin(θ), 0, np.cos(θ)]]), np.array([xc, 0, zc])
    pts = (pts - T) @ R # move and rotate points to the camera frame
    f = 2*np.tan(cam['fov']/2)*h/w# focal length
    ppts = - pts[:,1:] / pts[:,0:1] / f # project points onto the camera frame
    ppts = np.round(h*ppts + np.array([w//2, h//2])).astype(np.int32) # convert to pixel coordinates
    if points.ndim == 1: return ppts[0] #return a single point
    return ppts #return multiple points

            
            