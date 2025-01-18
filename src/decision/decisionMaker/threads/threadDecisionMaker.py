from src.decision.distance.distanceModule import DistanceModule
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CurrentSpeed, CurrentSteer, SetSpeed, SetSteer, SpeedMotor, SteerMotor, Ultra, mainCamera, CV_ObjectDetection_Type)
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
        self.subscribers = {}
        self.distanceModule = DistanceModule()
        self.speedSender = messageHandlerSender(self.queuesList, SetSpeed)
        self.steerSender = messageHandlerSender(self.queuesList, SetSteer)
        self.subscribe()
        super(threadDecisionMaker, self).__init__()
        self.ignore_stop_signal_until = 0
        self.previous_speed = 0

    def handle_stop_signal_logic(self, objectDetection, decidedSpeed):
        current_time = time.time()

        if objectDetection == "stop_signal" and current_time > self.ignore_stop_signal_until:
            self.speedSender.send("0")
            time.sleep(3)  # Esperar 3 segundos
            self.speedSender.send("40")
            self.ignore_stop_signal_until = current_time + 10  # Ignorar la señal de stop por 10 segundos

        return decidedSpeed
    
    def run(self):
        while self._running:
            ## Recieves the sub values
            ultraVals = self.subscribers["Ultra"].receive()
            objectDetection = self.subscribers["CV_ObjectDetection_Type"].receive()
            self.currentSpeed  = self.subscribers["CurrentSpeed"].receive() or self.currentSpeed 
            self.currentSteer  = self.subscribers["CurrentSteer"].receive() or self.currentSteer
            targetSpeed =  self.subscribers["SpeedMotor"].receive() or self.currentSpeed 
            targetSteer =  self.subscribers["SteerMotor"].receive() or self.currentSteer 
            # Decides speed based on distance safe check
            decidedSpeed, decidedSteer = self.distanceModule.check_distance(ultraVals, targetSpeed, targetSteer)
            decidedSpeed = self.handle_stop_signal_logic(objectDetection, decidedSpeed)
            #decidedSpeed = self.distanceModule.check_stop_signal(objectDetection, targetSpeed)
            # If there's change in steer or speed, sends the message to the nucleo board
            if self.currentSpeed != decidedSpeed:
                self.speedSender.send(decidedSpeed)
            if self.currentSteer != targetSteer:
                self.steerSender.send(decidedSteer)


    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, Ultra, "lastOnly", True)
        self.subscribers["Ultra"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, CV_ObjectDetection_Type, "lastOnly", True)
        self.subscribers["CV_ObjectDetection_Type"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, CurrentSpeed, "lastOnly", True)
        self.subscribers["CurrentSpeed"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, CurrentSteer, "lastOnly", True)
        self.subscribers["CurrentSteer"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, SpeedMotor, "lastOnly", True)
        self.subscribers["SpeedMotor"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, SteerMotor, "lastOnly", True)
        self.subscribers["SteerMotor"] = subscriber

