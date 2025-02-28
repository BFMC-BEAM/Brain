import datetime
import json
import os
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (ImuData, ImuGPSData)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

from src.hardware.imu_gps.imu_gps import imu_gps
import time

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
        self.timestamp = datetime.datetime.now().strftime("%d-%m-%H:%M")

        self.imu_gps_data = messageHandlerSender(self.queuesList, ImuGPSData)
        self.imu_data_history = []
        self.subscribe()
        super(threadimu_gps, self).__init__()
        self._running = True

        self.imu_gps = imu_gps()

        self.imu_data = []
        self.curr_coordinates = {
            # "x": 4,
            # "y": 0.75,
            # "yaw": 3.14,
            "x": 0,
            "y": 0,
            "yaw": 0,
        }


    def run(self):
        print("imu_gps thread is running")
        self.imu_gps.setInitialConditions(self.curr_coordinates)  # Set initial conditions

        last_time = time.time()  # Tiempo inicial

        while self._running:
            imuData = self.subscribers["ImuData"].receive()  # Get the imu data
            if imuData is not None:
                
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time  # Actualizar el último tiempo

                if self.logging is True: 
                    self.start_logging(imuData, dt)

                if self.debugging is True:
                     self.start_debugging(imuData, dt)

                time.sleep(0.1)
    def start_logging(self, imu_data, dt):
        # Asegurar que imu_data sea un diccionario
        if isinstance(imu_data, str):  
            imu_data = json.loads(imu_data)  # Convertir JSON a diccionario
        imu_data["dt"] = dt
        self.imu_data_history.append(imu_data)
        self.save_labels(self.imu_data_history,f'imu_history_{self.timestamp}.json')        

    def start_debugging(self, imu_data, dt):
        print(f'{imu_data},{dt}')

    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.subscribers["ImuData"] = subscriber        


        # Cargar labels
    def load_labels(self,labels_path):
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as file:
                return json.load(file)
        return {}
        

    def save_labels(self,labels, output_path):
        try:
            '''
            # Si ya existe el archivo de etiquetas, lo cargamos
            if os.path.exists(output_path):
                with open(output_path, 'r') as file:
                    existing_labels = json.load(file)
            else:
                existing_labels = {}
            
            # Actualizamos las etiquetas de las imágenes, sobreescribiendo las existentes
            existing_labels.update(labels)
            '''
            # Guardamos las etiquetas actualizadas en el archivo
            with open(output_path, 'w') as file:
                json.dump(labels, file, indent=4)
        
        except Exception as e:
            print(f"Error al guardar los labels: {e}")