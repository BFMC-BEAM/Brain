from src.decision.distance.distanceModule import DistanceModule
from src.decision.lineFollowing.purepursuit import ControlSystem
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CurrentSpeed, CurrentSteer, SetSpeed, SetSteer, SpeedMotor, SteerMotor, Ultra, mainCamera, Deviation, Direction, Lines, DrivingMode)
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
        self.prev_drivingMode = "stop"
        self.subscribe()
        super(threadDecisionMaker, self).__init__()

    def run(self):

        while self._running:
            ## Recieves the sub values
            ultraVals = self.subscribers["Ultra"].receive()
            direction = self.subscribers["Direction"].receive()
            self.currentSpeed  = self.subscribers["CurrentSpeed"].receive() or self.currentSpeed 
            self.currentSteer  = self.subscribers["CurrentSteer"].receive() or self.currentSteer
            targetSpeed =  self.subscribers["SpeedMotor"].receive() or self.currentSpeed 
            targetSteer =  self.subscribers["SteerMotor"].receive() or self.currentSteer
            new_deviation = self.subscribers["Deviation"].receive() or self.currentDeviation 
            new_lines = self.subscribers["Lines"].receive() or self.currentLines
            curr_drivingMode = self.subscribers["DrivingMode"].receive() or self.prev_drivingMode
            # Decides speed based on distance safe check
            decidedSpeed, decidedSteer = self.distanceModule.check_distance(ultraVals, targetSpeed, targetSteer)
                
            # If there's change in steer or speed, sends the message to the nucleo board

            # if self.currentSteer != decidedSteer:
            #     self.steerSender.send(decidedSteer)
            
            #if self.currentLines != new_lines:
             #   if new_lines == 2:
              #      self.speedSender.send("200")
               # elif new_lines == 1:
                #    self.speedSender.send("100")
                #self.currentLines = new_lines


            if self.currentSpeed != decidedSpeed:
                self.speedSender.send(decidedSpeed)
            if self.currentDeviation != new_deviation:
                new_steer = self.controlSystem.adjust_direction(new_deviation, direction)
                self.steerSender.send(str(new_steer * 10 )) # Revisar: new_steer llega al dashboard dividido por 10 ( new_steer=12 dashboard=1.2)
                self.currentDeviation = new_deviation

            if curr_drivingMode != self.prev_drivingMode :
                print(curr_drivingMode)
                self.prev_drivingMode = curr_drivingMode

            if curr_drivingMode is "auto":
                print("Auto")

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

    # =============================== START ===============================================
    def start(self):
        super(threadDecisionMaker, self).start()