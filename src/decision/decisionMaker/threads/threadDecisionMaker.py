from decision.distance.distanceModule import DistanceModule
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (CurrentSpeed, SpeedMotor, Ultra, mainCamera)
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
        self.currentSpeed = "0"
        self.subscribers = {}
        self.distanceModule = DistanceModule()
        self.speedSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.subscribe()
        super(threadDecisionMaker, self).__init__()

    def run(self):
        while self._running:
            ultraVals = self.subscribers["Ultra"].receive()
            self.currentSpeed  = self.subscribers["CurrentSpeed"].receive() or self.currentSpeed 
            targetSpeed, targetSteer = self.distanceModule.check_distance(ultraVals,self.currentSpeed)
            if targetSpeed is not None:
                self.speedSender.send(targetSpeed)

            

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, Ultra, "lastOnly", True)
        self.subscribers["Ultra"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, CurrentSpeed, "lastOnly", True)
        self.subscribers["CurrentSpeed"] = subscriber
