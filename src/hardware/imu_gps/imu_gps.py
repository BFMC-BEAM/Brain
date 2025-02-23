import numpy as np
import ast
from pykalman import KalmanFilter

gravity = 9.81  # m/s²

class imu_gps():
    def __init__(self, buffer_size=10):
        self.x = 0
        self.y = 0
        self.z = 0

        self.buffer_size = buffer_size
        self.accel_buffer_x = np.zeros(buffer_size)
        self.accel_buffer_y = np.zeros(buffer_size)
        self.buffer_index = 0

        self.kf = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],  # 2x2
            observation_matrices=[[1, 0]],         # 1x2
            initial_state_mean=[0, 0],             # Vector de tamaño 2
            initial_state_covariance=[[0.5, 0.0], [0.0, 0.5]],  # 2x2
            observation_covariance=[[0.05]],       # 1x1
            transition_covariance=[[0.01, 0.0], [0.0, 0.01]]    # 2x2
        )

        self.state_mean_x = np.array([0, 0])
        self.state_cov_x = np.eye(2)
        self.state_mean_y = np.array([0, 0])
        self.state_cov_y = np.eye(2)

    def setInitialConditions(self, initialData):
        self.x = initialData.get('x', 0)
        self.y = initialData.get('y', 0)
        self.yaw = initialData.get('yaw', 0)

    def getGpsData(self, imuData, delta_time):
        imuData = ast.literal_eval(imuData)
        roll = np.radians(float(imuData['roll']))
        pitch = np.radians(float(imuData['pitch']))
        yaw = np.radians(float(imuData['yaw']))
        accelX = float(imuData['accelx'])
        accelY = float(imuData['accely'])
        accelZ = float(imuData['accelz'])

        # Compensación de gravedad
        gravity_comp_x = np.sin(pitch) * gravity
        gravity_comp_y = -np.sin(roll) * gravity

        # Ajuste por yaw
        adjusted_accelX = accelX * np.cos(yaw) - accelY * np.sin(yaw)
        adjusted_accelY = accelX * np.sin(yaw) + accelY * np.cos(yaw)

        # Aplicar filtro de Kalman
        self.state_mean_x, self.state_cov_x = self.kf.filter_update(
            self.state_mean_x, self.state_cov_x, observation=adjusted_accelX
        )
        self.state_mean_y, self.state_cov_y = self.kf.filter_update(
            self.state_mean_y, self.state_cov_y, observation=adjusted_accelY
        )

        # Integración para actualizar posición
        vel_x = self.state_mean_x[1] * delta_time
        vel_y = self.state_mean_y[1] * delta_time
        vel_x = adjusted_accelX * delta_time
        vel_y = adjusted_accelY * delta_time

        self.x += vel_x * delta_time
        self.y += vel_y * delta_time

        return {
            "x": self.x,
            "y": self.y,
        }
