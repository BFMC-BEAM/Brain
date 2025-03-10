import time
from src.decision.distance.distanceModule import DistanceModule
from src.decision.lineFollowing.purepursuit import ControlSystem
from src.templates.threadwithstop import ThreadWithStop
from src.decision.decisionMaker.stateMachine import StateMachine
from src.utils.constants import (
        # ========================= SIGNS ==========================
        STOP_SIGN,
        PARKING_SIGN ,
        CROSSWALK_SIGN,
        PRIORITY_SIGN,
        HIGHWAY_ENTRY_SIGN,
        HIGHWAY_END_SIGN,
        ONE_WAY_SIGN,
        ROUNDABOUT_SIGN,
        NO_ENTRY_SIGN,
        # ========================= OBSTACLES ==========================
        PEDESTRIAN,
        CAR,
        ROADBLOCK)
import numpy as np

from src.utils.messages.allMessages import (
    CurrentSpeed, 
    CurrentSteer, 
    SetSpeed, 
    SetSteer, 
    SpeedMotor, 
    SteerMotor,
    StopLineDistance, 
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

        self.signals = [
            STOP_SIGN,
            PARKING_SIGN ,
            CROSSWALK_SIGN,
            PRIORITY_SIGN,
            HIGHWAY_ENTRY_SIGN,
            HIGHWAY_END_SIGN,
            ONE_WAY_SIGN,
            ROUNDABOUT_SIGN,
            NO_ENTRY_SIGN
        ]
        self.obstacles = [
            PEDESTRIAN,
            CAR,
            ROADBLOCK
        ]

        self.current_speed = 100
        self.current_steer = "0"
        self.current_direction = None
        self.current_deviation = 0.
        self.objects_detected = None
        self.intersection = -1
        self.stopline_distance = None

    def run(self):

        while self._running:

            curr_drivingMode = self.subscribers["DrivingMode"].receive() or self.curr_drivingMode
            self.curr_drivingMode = curr_drivingMode

            if curr_drivingMode == "auto":
                self.current_deviation = self.subscribers["Deviation"].receive() or self.current_deviation
                self.current_direction = self.subscribers["Direction"].receive() or self.current_direction
                self.objects_detected = self.subscribers["CV_ObjectsDetected"].receive()
                self.current_speed  = self.subscribers["CurrentSpeed"].receive() or self.current_speed
                self.current_steer  = self.subscribers["CurrentSteer"].receive() or self.current_steer
                ultra_values = self.subscribers["Ultra"].receive()          
                self.stopline_distance = self.subscribers["StopLineDistance"].receive()          
                self.current_speed = float(self.current_speed)
                signs_detected = []
                obstacles_detected = []
                
                if self.stopline_distance:
                    print(self.stopline_distance)
                
                if self.objects_detected:
                    decoded_objects = self._decode(self.objects_detected)
                    for class_name, foward_distance, lateral_distance in decoded_objects:

                        if class_name in self.signals:
                            if foward_distance < 30:
                                print(class_name, foward_distance)

                                signs_detected.append((class_name, True, lateral_distance))
                            else:
                                signs_detected.append((class_name, False, lateral_distance))
                        elif class_name in self.objects_detected:
                            obstacles_detected.append((class_name, foward_distance, lateral_distance))
                
            
                ultra_values = 0

                target_speed, target_steer = self.state_machine.handle_events(
                    self.current_deviation, self.current_direction, 0, self.current_speed, signs_detected, obstacles_detected, self.stopline_distance, ultra_values
                )
                
                target_speed = str(target_speed)
                target_steer = str(target_steer)
                if target_speed != self.current_speed:
                    #print("Devuelve: ", target_speed)

                    #TODO: no deberia entrar varias veces con misma velocidad
                    self.speedSender.send(target_speed)
                    self.current_speed = target_speed

                if target_steer != self.current_steer:
                    self.steerSender.send(target_steer)
                    self.current_steer = target_steer

            elif curr_drivingMode == "manual":
                target_speed =  self.subscribers["SpeedMotor"].receive() 
                target_steer =  self.subscribers["SteerMotor"].receive() 

                if target_speed is not None:
                    self.speedSender.send(str(target_speed))
                if target_steer is not None:
                    self.steerSender.send(str(target_steer))
            elif curr_drivingMode == "stop":
                self.speedSender.send("0")
                self.steerSender.send("0")

            time.sleep(0.03)

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

        subscriber = messageHandlerSubscriber(self.queuesList, StopLineDistance, "fifo", True)
        self.subscribers["StopLineDistance"] = subscriber
        

    # =============================== START ===============================================
    def start(self):
        super(threadDecisionMaker, self).start()