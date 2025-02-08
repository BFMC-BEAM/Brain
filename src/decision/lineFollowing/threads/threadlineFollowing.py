from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (mainCamera, Deviation, Direction)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
class threadlineFollowing(ThreadWithStop):
    """This thread handles lineFollowing.
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
        super(threadlineFollowing, self).__init__()

    def run(self):
        while self._running:
            print("Deviation: ", self.subscribers["Deviation"].receive())
            print("Direction: ", self.subscribers["Direction"].receive())

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        
        subscriber = messageHandlerSubscriber(self.queuesList, Deviation, "lastOnly", True)
        self.subscribers["Deviation"] = subscriber
        subscriber = messageHandlerSubscriber(self.queuesList, Direction, "lastOnly", True)
        self.subscribers["Direction"] = subscriber
