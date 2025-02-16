from src.decision.distance.distanceModule import DistanceModule
from src.decision.lineFollowing.purepursuit import ControlSystem
from src.templates.threadwithstop import ThreadWithStop
from src.decision.decisionMaker.stateMachine import StateMachine
from src.ComputerVision.LaneDetection.lane_detection import LaneDetectionProcessor
from src.ComputerVision.ObjectDetection.object_detection import ObjectDetectionProcessor
import cv2
import base64
import numpy as np

from src.utils.messages.allMessages import (
    CurrentSpeed, 
    CurrentSteer, 
    SetSpeed, 
    SetSteer, 
    SpeedMotor, 
    SteerMotor, 
    Ultra, 
    CVCamera, 
    CVCameraProcessed,
    serialCamera,
    CV_ObjectDetection_Type,
    Deviation, 
    Direction, 
    Lines, 
    DrivingMode)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

import time

class threadDecisionMaker(ThreadWithStop):
    """This thread handles decisionMaker.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.currentSpeed = "0"
        self.currentSteer = "0"
        self.currentDeviation = 0.
        self.currentLines = -1
        self.subscribers = {}
        self.distanceModule = DistanceModule()
        self.controlSystem = ControlSystem()
        self.speedSender = messageHandlerSender(self.queuesList, SetSpeed)
        self.steerSender = messageHandlerSender(self.queuesList, SetSteer)
        self.lines = messageHandlerSender(self.queuesList, Lines) #TODO: modificar nombre
        self.serialCameraSender = messageHandlerSender(self.queuesList, CVCamera)
        self.lane_processor = LaneDetectionProcessor(type="simulator")
        self.processor = ObjectDetectionProcessor()
        self.act_lines = -1 # contador de lineas detectadas, 0 nada, 1 si detecto izq o der, 2 normal
        #self.state_machine = StateMachine(self.lane_processor, self.processor, self.direction, self.deviation, self.ObjectDetection_Type, self.lines )
        self.prev_drivingMode = "stop"
        self.subscribe()
        super(threadDecisionMaker, self).__init__()


    def run(self):

        while self._running:

            ## Recieves the sub values
            #  Serial communication
            self.currentSpeed  = self.subscribers["CurrentSpeed"].receive() or self.currentSpeed 
            self.currentSteer  = self.subscribers["CurrentSteer"].receive() or self.currentSteer
            targetSpeed =  self.subscribers["SpeedMotor"].receive() or self.currentSpeed 
            targetSteer =  self.subscribers["SteerMotor"].receive() or self.currentSteer
            ultraVals = self.subscribers["Ultra"].receive()
            # Object Detection
            distance = self.subscribers["CV_ObjectDetection_Type"].receive() or 777
            # Lines
            new_deviation = self.subscribers["Deviation"].receive() or self.currentDeviation 
            new_direction = self.subscribers["Direction"].receive()
            new_lines = self.subscribers["Lines"].receive() or self.currentLines
            # Dashboard
            new_drivingMode = self.subscribers["DrivingMode"].receive() or self.prev_drivingMode
            
            # FrameCamera = self.subscribers["serialCamera"].receive() 

            # if FrameCamera is None:
            #     continue
            
            # #print(FrameCamera)
            # decoded_image_data = base64.b64decode(FrameCamera)
            
            # # Paso 2: Convertir los bytes en una imagen
            # nparr = np.frombuffer(decoded_image_data, np.uint8)
            # FrameCamera = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # # Decides speed based on distance safe check
            # #self.state_machine.run(FrameCamera)
            # FrameCameraPro=self.lane_processor.process_image(FrameCamera)
            
            # _, serialEncodedImg = cv2.imencode(".jpg", FrameCameraPro)#self.state_machine.frame)

            # serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8")
            # self.serialCameraSender.send(serialEncodedImageData)
            
            # decidedSpeed, decidedSteer = self.distanceModule.check_distance(ultraVals, targetSpeed, targetSteer)

            # # If there's change in steer or speed, sends the message to the nucleo board

            # # if self.currentSteer != decidedSteer:
            # #     self.steerSender.send(decidedSteer)
           
            # if self.currentLines != new_lines:
            #     if new_lines == 2:
            #         self.speedSender.send("200")
            #     elif new_lines == 1:
            #         self.speedSender.send("100")
            #     self.currentLines = new_lines
            # # Object Detection
            # distance = self.subscribers["CV_ObjectDetection_Type"].receive() or 777
            # # Dashboard
            # new_drivingMode = self.subscribers["DrivingMode"].receive() or self.prev_drivingMode


            # Ultrasound Logic
            print("----------Ultrasound----------")
            print("UltraVals: ", ultraVals)
            # decidedSpeed, decidedSteer = self.distanceModule.check_distance(ultraVals, targetSpeed, targetSteer)

            # Object Detection Logic

            print("----------Object detection----------")
            print("Distance: ", distance)

            # Lane Detection Logic 

            print("----------Lane detection----------")
            print("Deviation: ", new_deviation)
            print("Direction: ", new_direction)
            print("Lines: ", new_lines)

            # if self.currentDeviation != new_deviation:
            #     new_steer = self.controlSystem.adjust_direction(new_deviation, direction)
            #     self.steerSender.send(str(new_steer * 10 )) # Revisar: new_steer llega al dashboard dividido por 10 ( new_steer=12 dashboard=1.2)
            #     self.currentDeviation = new_deviation


            # Dashboard

            print("Driving Mode: ", new_drivingMode)
            # if new_drivingMode != self.prev_drivingMode:
            #     self.prev_drivingMode = new_drivingMode
            #     print("Driving Mode: ", new_drivingMode)


            time.sleep(0.2)  # Pausa la ejecución por 2 segundos


            

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, Ultra, "lastOnly", True)
        self.subscribers["Ultra"] = subscriber

        subscriber = messageHandlerSubscriber(self.queuesList, Deviation, "lastOnly", True)
        self.subscribers["Deviation"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, Direction, "lastOnly", True)
        self.subscribers["Direction"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, Lines, "lastOnly", True)
        self.subscribers["Lines"] = subscriber

        subscriber  = messageHandlerSubscriber(self.queuesList, CV_ObjectDetection_Type, "lastOnly", True)
        self.subscribers["CV_ObjectDetection_Type"] = subscriber

        subscriber = messageHandlerSubscriber(self.queuesList, CurrentSpeed, "lastOnly", True)
        self.subscribers["CurrentSpeed"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, CurrentSteer, "lastOnly", True)
        self.subscribers["CurrentSteer"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, SpeedMotor, "lastOnly", True)
        self.subscribers["SpeedMotor"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, SteerMotor, "lastOnly", True)
        self.subscribers["SteerMotor"] = subscriber

        subscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.subscribers["DrivingMode"] = subscriber

        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.subscribers["serialCamera"] = subscriber

        subscriber = messageHandlerSubscriber(self.queuesList, CVCameraProcessed, "lastOnly", True)
        self.subscribers["CVCameraProcessed"] = subscriber

    # =============================== START ===============================================
    def start(self):
        super(threadDecisionMaker, self).start()