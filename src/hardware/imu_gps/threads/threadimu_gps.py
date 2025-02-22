from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (mainCamera, ImuData)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

from src.hardware.imu_gps.imu_gps import imu_gps

##########################
# Initial coordinates    #
# Paso de peatones       #
# x: 4 y: 0.75 yaw: 3.14 #
##########################


class threadimu_gps(ThreadWithStop):
    """This thread handles imu_gps.
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
        super(threadimu_gps, self).__init__()

        self.imu_gps = imu_gps()

        self.curr_coordinates = {
            "x": 4,
            "y": 0.75,
            "yaw": 3.14,
        }


    def run(self):
        print("imu_gps thread is running")
        self.imu_gps.setInitialCoditions(self.curr_coordinates) # Set initial conditions
        
        while self._running:
            imuData = self.subscribers["ImuData"].receive() # Get the imu data
            self.curr_coordinates = self.imu_gps.getGpsData(imuData) # Get the gps data from imu
            print("Current coordinates: ", self.curr_coordinates)

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.subscribers["ImuData"] = subscriber        
