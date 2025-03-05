import datetime
import numpy as np
import cv2
import queue
import base64
import matplotlib.pyplot as plt
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (ImuData, serialCamera, MapImage)
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

        self.velocity = np.array([0.0, 0.0])
        self.imu_position = np.array([0.0, 0.0])
        self.positions = []  # Lista para almacenar la trayectoria

        self.optical_flow = OpticalFlowOdometry(scale_factor=1/0.6 )  
        self.subscribe()
        super(threadimu_gps, self).__init__()

        self.cam_position = [4,0.75]

        # Inicializa el filtro de Kalman
        process_noise_cov = np.eye(4) * 0.00000000005  
        measurement_noise_cov = np.eye(4) * 500  
        initial_error_cov = np.eye(4) * 0.05  
        
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
            frameData = self.subscribers["serialCamera"].receive()

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

                self.velocity = np.array([velocity[0], velocity[1]])  
                self.imu_position = np.array([position[0]*10, position[1]*10])  

                # print(f"IMU: {self.imu_position}")
                # self.imu_gps_data.send(self.imu_position)
                #Dibujar mapa

                self.imu_position[0] = (self.imu_position[0]/10)+4
                self.imu_position[1] = (self.imu_position[1]/10)+0.75

            if frameData is not None:
                frame_bytes = base64.b64decode(frameData)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    self.cam_position = self.optical_flow.process_frame(frame)
                    # print(f"Camera: {cam_position}")
                    combined_position = (w_imu * self.imu_position) + (w_cam * self.cam_position[:2])
                    
                    # Guardar posición en la lista
                    self.positions.append(combined_position)

                    # Graficar y guardar imagen
                    self.plot_positions()
                    
                    print(f"Posición combinada (IMU + Cámara): {combined_position} \t|\t IMU: {self.imu_position} \t|\t Camera: {self.cam_position}")

                    # self.map_drawer.add_gps_data(combined_position[0], combined_position[1])
                    # drawn_map = self.map_drawer.get_current_map()
                    # self.map_image_sender.send(encode_image(drawn_map))

            time.sleep(0.01)

    def plot_positions(self):
        """Genera y guarda una imagen con solo el punto actual"""
        if not self.positions:
            return  # No hay posiciones registradas

        x, y = self.positions[-1]  # Última posición

        plt.figure(figsize=(6, 6), dpi=100)
        plt.scatter(x, y, color='red', s=100)  # Dibuja solo el punto en rojo
        plt.xlabel("Posición X")
        plt.ylabel("Posición Y")
        plt.title("Posición actual del auto RC")
        plt.grid(True)

        plt.xlim(-6, 6)  # Mantener escala fija
        plt.ylim(-6, 6)

        plt.savefig("posicion.png", bbox_inches="tight")
        plt.close()


    def subscribe(self):
        """Suscribirse a los mensajes de IMU y cámara"""
        self.subscribers["ImuData"] = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.subscribers["serialCamera"] = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
