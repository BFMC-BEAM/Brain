import datetime
import numpy as np
import cv2
import queue
import base64
import matplotlib.pyplot as plt
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (ImuData, serialCamera, MapImage, CurrentSteer, CurrentSpeed)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import ast
from src.hardware.imu_gps.opticalFlowOdometry import OpticalFlowOdometry
from src.hardware.imu_gps.kalman_filter_imu import KalmanFilterIMU

import json
from src.hardware.imu_gps.map_drawing import MapDrawer
from src.templates.threadwithstop import ThreadWithStop
import networkx as nx
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
    """Hilo que maneja la IMU y procesa la odometría"""
    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.subscribers = {}
        self.timestamp = datetime.datetime.now().strftime("%d-%m-%H:%M")

        self.imu_gps_data = messageHandlerSender(self.queuesList, ImuData)
        self.imu_data_history = []
        
        self.track_graph = nx.read_graphml(TRACK_GRAPH_PATH)
        self.map_drawer = MapDrawer(self.track_graph)
        self.map_image_sender = messageHandlerSender(self.queuesList, MapImage)

        self.acceleration = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.imu_position = np.array([4.0, 0.75])
        self.positions = []  # Lista para almacenar la trayectoria

        self.optical_flow = OpticalFlowOdometry(scale_factor=1/0.6 )  
        self.subscribe()
        super(threadimu_gps, self).__init__()

        self.cam_position = [4,0.75]
        self.count = 0
        # Inicializa el filtro de Kalman
        process_noise_cov = np.eye(4) * 0.01  
        measurement_noise_cov = np.eye(4) * 0.1  
        initial_error_cov = np.eye(4) * 0.01  
        

        self.values = []  # Lista para almacenar los valores
        self.num_samples = 50  # Número de muestras a promediar

        self.kalman_filter = KalmanFilterIMU(dt=0.15,
                                            process_noise_cov=process_noise_cov,
                                            measurement_noise_cov=measurement_noise_cov,
                                            initial_error_cov=initial_error_cov)  

    def run(self):
        self._running = True
        print("IMU y odometría thread está corriendo")

        w_imu = 0
        w_cam = 1
        
        while self._running:
            imuData = self.subscribers["ImuData"].receive()
            # frameData = self.subscribers["serialCamera"].receive()

            if imuData is not None:
                imuData = ast.literal_eval(imuData)

                acceleration = np.array([
                    float(imuData["accelx"]),
                    float(imuData["accely"]),
                    float(imuData["accelz"])
                ])

                velocity = np.array([
                    float(imuData["velx"]), 
                    float(imuData["vely"]), 
                    float(imuData["velz"])
                ])
                position = np.array([
                    float(imuData["posx"]), 
                    float(imuData["posy"]), 
                    float(imuData["posz"])
                ])

                print(acceleration, velocity, position)

                 # self.acceleration = np.array([acceleration[0],acceleration[1]])
                # self.velocity = np.array([velocity[0], velocity[1]])  
                self.imu_position = np.array([ (velocity[0]*1.25/100+4),(-velocity[1]*1.25/100)+0.75])  
                
                self.velocity = np.array([velocity[2]])  
                
                # self.imu_gps_data.send(self.imu_position)
                # Dibujar mapa
                # if self.count > 50 :
                self.map_drawer.add_gps_data(self.imu_position[0], self.imu_position[1])
                drawn_map = self.map_drawer.get_current_map()
                self.map_image_sender.send(encode_image(drawn_map))
                self.count=0

                print(f"IMU: {self.imu_position}")
                print(f"counter: {self.velocity}")
                # rpi_position = self.kalman_filter.update(self.acceleration)

            time.sleep(0.5)


    def subscribe(self):
        """Suscribirse a los mensajes de IMU y cámara"""
        self.subscribers["ImuData"] = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        #self.subscribers["serialCamera"] = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        # self.subscribers["CurrentSpeed"] = messageHandlerSubscriber(self.queuesList, CurrentSpeed, "lastOnly", True)
        # self.subscribers["CurrentSteer"] = messageHandlerSubscriber(self.queuesList, CurrentSteer, "lastOnly", True)
