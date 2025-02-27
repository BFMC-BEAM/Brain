import numpy as np
import ast
from pykalman import KalmanFilter

class imu_gps():
    def __init__(self, buffer_size=5):
        self.x = 0
        self.y = 0
        self.z = 0

        self.buffer_size = buffer_size
        self.vel_buffer_x = np.zeros(buffer_size)
        self.vel_buffer_y = np.zeros(buffer_size)
        self.buffer_index = 0

        # Variables para manejar el yaw acumulado
        self.last_yaw = None
        self.total_yaw = 0  # Rotación acumulada en grados

        self.kf_x = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],  # Modelo de velocidad constante
            observation_matrices=[[1, 0]],         # Observamos solo la velocidad
            initial_state_mean=[0, 0],             # Estado inicial: [posición, velocidad]
            initial_state_covariance=[[0.1, 0.0], [0.0, 0.1]],  # Incertidumbre inicial
            observation_covariance=[[1.0]],        # Confianza en las mediciones
            transition_covariance=[[0.001, 0.0], [0.0, 0.001]]  # Suavizado
        )

        self.kf_y = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],  # Modelo de velocidad constante
            observation_matrices=[[1, 0]],         # Observamos solo la velocidad
            initial_state_mean=[0, 0],             # Estado inicial: [posición, velocidad]
            initial_state_covariance=[[0.1, 0.0], [0.0, 0.1]],  # Incertidumbre inicial
            observation_covariance=[[1.0]],        # Confianza en las mediciones
            transition_covariance=[[0.001, 0.0], [0.0, 0.001]]  # Suavizado
        )

        self.state_mean_x = np.array([0, 0])
        self.state_cov_x = np.eye(2)
        self.state_mean_y = np.array([0, 0])
        self.state_cov_y = np.eye(2)

    def setInitialConditions(self, initialData):
        self.x = initialData.get('x', 0)
        self.y = initialData.get('y', 0)
        self.yaw = initialData.get('yaw', 0)

    # angle unwrapping
    def update_yaw(self, raw_yaw):
        yaw = float(raw_yaw)

        if self.last_yaw is not None:
            delta = yaw - self.last_yaw

            # Detectar wrap-around
            if delta > 30:
                delta -= 60
            elif delta < -30:
                delta += 60

            self.total_yaw += delta  # Acumular la rotación total

        self.last_yaw = yaw
        return (self.total_yaw % 360)

    def getGpsData(self, imuData, delta_time):
        imuData = ast.literal_eval(imuData)

        # Actualizar yaw acumulado
        yaw_degrees = self.update_yaw(imuData['yaw'])
        yaw = np.radians(yaw_degrees)  # Convertir a radianes
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # Normalizar el ángulo al rango [-pi, pi]

        # Velocidades
        velX = float(imuData['velx'])
        velY = float(imuData['vely'])

        # Actualizar los buffers de velocidad (buffer circular)
        self.vel_buffer_x[self.buffer_index] = velX
        self.vel_buffer_y[self.buffer_index] = velY
        
        # Actualizar el índice de manera circular
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size

        # Calcular el promedio de las velocidades almacenadas en el buffer
        adjusted_velX = np.mean(self.vel_buffer_x)
        adjusted_velY = np.mean(self.vel_buffer_y)

        # # Aplicar el filtro de Kalman a las velocidades
        # self.state_mean_x, self.state_cov_x = self.kf_x.filter_update(
        #     self.state_mean_x, self.state_cov_x, observation=adjusted_velX
        # )
        # self.state_mean_y, self.state_cov_y = self.kf_y.filter_update(
        #     self.state_mean_y, self.state_cov_y, observation=adjusted_velY
        # )

        # # Obtener las velocidades filtradas
        # filtered_velX = self.state_mean_x[0]
        # filtered_velY = self.state_mean_y[0]

        # Ajuste por yaw acumulado
        delta_d = adjusted_velY * delta_time

        # Actualizar posición usando las velocidades filtradas
        self.x += delta_d * np.sin(yaw)
        self.y += delta_d * np.cos(yaw)

        print(f"x: {self.x:.3f} y: {self.y:.3f} \t|\t yaw: {yaw:.3f} \t|\t velx: {adjusted_velX:.3f}, vely: {adjusted_velY:.3f}")

        return {
            "x": self.x,
            "y": self.y,
        }