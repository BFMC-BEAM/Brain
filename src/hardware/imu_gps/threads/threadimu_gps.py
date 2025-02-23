from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (ImuData)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

from src.hardware.imu_gps.imu_gps import imu_gps
import time
import ast  # Para convertir string en diccionario
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
        self.subscribers = {}
        self.subscribe()
        super(threadimu_gps, self).__init__()
        self._running = True

        self.imu_gps = imu_gps()

        self.imu_data = []
        self.curr_coordinates = {
            "x": 4,
            "y": 0.75,
            "yaw": 3.14,
        }


    def run(self):
        print("imu_gps thread is running")
        self.imu_gps.setInitialConditions(self.curr_coordinates)  # Set initial conditions

        last_time = time.time()  # Tiempo inicial

        while self._running:
            imuData = self.subscribers["ImuData"].receive()  # Get the imu data
            if imuData is not None:
                # print(imuData)
                # Calcular delta de tiempo real
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time  # Actualizar el último tiempo

                # Actualizar coordenadas usando el delta de tiempo dinámico
                self.curr_coordinates = self.imu_gps.getGpsData(imuData, dt)
                print("Current coordinates: ", self.curr_coordinates)

                time.sleep(0.15)  # Pequeña pausa para no saturar el CPU

    
    def check_distance(self, ultraVals, currentSpeed, currentSteer):
            mult_distance = self.min_distance * self.get_multiplier(currentSpeed)
            if ultraVals is not None:
                if ultraVals["top"] < mult_distance and int(currentSpeed) > 0:
                    return ("0",currentSteer) #stop the vehicle if front distance is less than 30 cm
                #elif ultraVals["bottom"] < self.min_distance and  int(currentSpeed) < 0:
                    # return ("0",currentSteer) commented until back ultra instalation
                    #pass
            return (currentSpeed,currentSteer)
    
    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.subscribers["ImuData"] = subscriber        