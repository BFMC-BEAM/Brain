import json
from src.hardware.imu_gps.map_drawing import MapDrawer
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (ImuData, ImuGPSData, MapImage)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import networkx as nx
from src.hardware.imu_gps.imu_gps import imu_gps
import time

from src.utils.helpers import encode_image
from src.utils.constants import GPS_DATA_PATH, TRACK_GRAPH_PATH
AVG_FRAME_COUNT = 10



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
        self.track_graph = nx.read_graphml(TRACK_GRAPH_PATH)
        self.map_drawer = MapDrawer(self.track_graph)
        self.imu_gps_data = messageHandlerSender(self.queuesList, ImuGPSData)
        self.map_image_sender = messageHandlerSender(self.queuesList, MapImage)
        self.gps_x = []
        self.gps_y = []


        self.subscribe()
        super(threadimu_gps, self).__init__()
        self._running = True

        self.imu_gps = imu_gps()

        self.imu_data = []
        self.curr_coordinates = {
            "x": 4,
            "y": 0.75,
            "yaw": 3.14,
            #"x": 0,
            #"y": 0,
            #"yaw": 0,
        }
        self.coord_history = []


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

                # {'x' : 0, 'y': 0}
                self.curr_coordinates = self.imu_gps.getGpsData(imuData, dt)    # Actualizar coordenadas usando el delta de tiempo dinámico
                # self.imu_gps_data.send(self.curr_coordinates)                   # Enviar las coordenadas
                # print("Current coordinates: ", self.curr_coordinates, imuData)

                self.coord_history.append(self.curr_coordinates)
                #Dibujar mapa
                self.map_drawer.add_gps_data(self.curr_coordinates.x, self.curr_coordinates.y)
                drawn_map = self.map_drawer.get_current_map()
                self.map_image_sender.send(encode_image(drawn_map))
            time.sleep(0.15)  # Pequeña pausa para no saturar el CPU
            
 
    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.subscribers["ImuData"] = subscriber        