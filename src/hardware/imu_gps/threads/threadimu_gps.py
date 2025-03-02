from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (ImuData, ImuGPSData, serialCamera)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from filterpy.kalman import KalmanFilter
import numpy as np
import time
import cv2
from pyslam.slam import SLAMSystem  # Importamos PySLAM

class threadimu_gps(ThreadWithStop):
    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.subscribers = {}

        self.imu_gps_data = messageHandlerSender(self.queuesList, ImuGPSData)

        self.subscribe()
        self.init_kalman()  # Inicializar Filtro de Kalman
        self.init_pyslam()   # Inicializar PySLAM
        super(threadimu_gps, self).__init__()
        self._running = True

    def init_kalman(self):
        """Inicializa el Filtro de Kalman para fusionar IMU"""
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  # 6 estados (x, y, z, vx, vy, vz) y 3 observaciones (IMU)
        self.kf.F = np.eye(6)  # Matriz de transición
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])
        self.kf.P *= 1000  # Covarianza inicial grande
        self.kf.R *= 5  # Ruido de medición

    def update_kalman(self, imu_data):
        """Actualiza el Filtro de Kalman con datos del IMU"""
        acceleration = np.array([imu_data.ax, imu_data.ay, imu_data.az])  # Extraer aceleraciones
        self.kf.predict()
        self.kf.update(acceleration)
        return self.kf.x[:3]  # Retorna la posición estimada

    def init_pyslam(self):
        """Inicializa el sistema de SLAM"""
        self.slam = SLAMSystem()  # Instancia PySLAM

    def run(self):
        print("imu_gps thread is running")
        imu_buffer = []  # Para almacenar datos recientes del IMU

        while self._running:
            imu_data = self.subscribers["ImuData"].receive()
            frame_camera = self.subscribers["serialCamera"].receive()

            if imu_data is not None:
                timestamp_imu = time.time()
                imu_buffer.append((timestamp_imu, imu_data))
                if len(imu_buffer) > 10:  # Mantén solo los últimos 10 datos
                    imu_buffer.pop(0)

                print(f"IMU Data -> Acc: ({imu_data.ax:.2f}, {imu_data.ay:.2f}, {imu_data.az:.2f}) "
                    f"Gyr: ({imu_data.gx:.2f}, {imu_data.gy:.2f}, {imu_data.gz:.2f})")

            if frame_camera is not None:
                timestamp_cam = time.time()

                # Buscar el dato de IMU más cercano en tiempo
                if imu_buffer:
                    closest_imu = min(imu_buffer, key=lambda t: abs(t[0] - timestamp_cam))
                    imu_data = closest_imu[1]

                    # Estimar posición con el IMU
                    estimated_position = self.update_kalman(imu_data)

                    print(f"Kalman Estimated Position -> X: {estimated_position[0]:.2f}, "
                        f"Y: {estimated_position[1]:.2f}, Z: {estimated_position[2]:.2f}")

                    # Convertir imagen a escala de grises
                    frame_gray = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2GRAY)

                    # Enviar datos a PySLAM
                    self.slam.process_frame(frame_gray, estimated_position)

                    # Obtener la pose estimada del SLAM
                    slam_pose = self.slam.get_current_position()

                    print(f"SLAM Position -> X: {slam_pose[0]:.2f}, Y: {slam_pose[1]:.2f}, Z: {slam_pose[2]:.2f}")

                    # Enviar la pose fusionada
                    self.imu_gps_data.send(slam_pose)

            time.sleep(0.1)  # Ajusta el tiempo de espera según sea necesario


    
    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        subscriber = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.subscribers["ImuData"] = subscriber     
        subscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "fifo", True)
        self.subscribers["serialCamera"] = subscriber    