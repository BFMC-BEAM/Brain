import time
from src.decision.distance.distanceModule import DistanceModule
from src.decision.lineFollowing.purepursuit import ControlSystem
from src.templates.threadwithstop import ThreadWithStop
from src.decision.decisionMaker.stateMachine import StateMachine
import numpy as np

from src.utils.messages.allMessages import (
    CurrentSpeed, 
    CurrentSteer, 
    SetSpeed, 
    SetSteer, 
    SpeedMotor, 
    SteerMotor, 
    Ultra,
    CV_ObjectsDetected,
    Deviation, 
    Direction, 
    Lines, 
    DrivingMode)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender


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
        self.subscribers = {}
        self.speedSender = messageHandlerSender(self.queuesList, SetSpeed)
        self.steerSender = messageHandlerSender(self.queuesList, SetSteer)
        self.deviation = messageHandlerSender(self.queuesList, Deviation)
        self.direction = messageHandlerSender(self.queuesList, Direction)
        self.lines = messageHandlerSender(self.queuesList, Lines) #TODO: modificar nombre
        self.curr_drivingMode = "stop"
        self.state_machine = StateMachine()
        self.subscribe()
        super(threadDecisionMaker, self).__init__()

        self.current_speed = 100
        self.current_steer = "0"
        self.current_direction = None
        self.current_deviation = 0.
        self.objects_detected = None
        self.intersection = -1

    def run(self):

        while self._running:

            curr_drivingMode = self.subscribers["DrivingMode"].receive() or self.curr_drivingMode
            self.curr_drivingMode = curr_drivingMode

            if curr_drivingMode == "auto":
                self.current_deviation = self.subscribers["Deviation"].receive() or self.current_deviation
                self.objects_detected = self.subscribers["CV_ObjectsDetected"].receive() or self.objects_detected
                self.current_speed  = self.subscribers["CurrentSpeed"].receive() or self.current_speed
                self.current_steer  = self.subscribers["CurrentSteer"].receive() or self.current_steer
                self.direction = self.subscribers["Direction"].receive() or self.current_direction
                ultra_values = self.subscribers["Ultra"].receive()          
                print(self.objects_detected)
                signs_detected = []
                obstacles_detected = []
                ultra_values = 0
                stopline_valid_distance = 0

                target_speed, target_steer = self.state_machine.handle_events(
                    self.current_deviation, self.current_direction, 0, self.current_speed, signs_detected, obstacles_detected, stopline_valid_distance, ultra_values
                )
                
                target_speed = str(target_speed)
                target_steer = str(target_steer)
                if target_speed != self.current_speed:
                    print("enviando velocidad: ", target_speed)
                    self.speedSender.send(target_speed)
                    self.current_speed = target_speed

                if target_steer != self.current_steer:
                    self.steerSender.send(target_steer)
                    self.current_steer = target_steer

            elif curr_drivingMode == "manual":
                target_speed =  self.subscribers["SpeedMotor"].receive() 
                target_steer =  self.subscribers["SteerMotor"].receive() 

                if (target_speed != None):
                    print("manual vel: ", target_speed)
                if (target_steer != None):
                    print("manual steer: ", target_steer)

                if target_speed is not None:
                    self.speedSender.send(str(target_speed))
                if target_steer is not None:
                    self.steerSender.send(str(target_steer))

            # elif curr_drivingMode == "stop":
            #     self.speedSender.send("0")
            #     self.steerSender.send("0")
            time.sleep(0.03)

            

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
        subscriber = messageHandlerSubscriber(self.queuesList, Lines, "lastOnly", True)
        self.subscribers["Intersection"] = subscriber

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

        subscriber = messageHandlerSubscriber(self.queuesList, CV_ObjectsDetected, "lastOnly", True)
        self.subscribers["CV_ObjectsDetected"] = subscriber
        

    # =============================== START ===============================================
    def start(self):
        super(threadDecisionMaker, self).start()