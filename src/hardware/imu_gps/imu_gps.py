import numpy as np
import ast

# Compensar la gravedad en el eje Z
gravity = 9.81  # m/s²


class imu_gps():
    def __init__(self, buffer_size=30):
        self.x = 0
        self.y = 0
        self.z = 0

        self.buffer_size = buffer_size
        self.accel_buffer_x = np.zeros(buffer_size)
        self.accel_buffer_y = np.zeros(buffer_size)
        self.accel_buffer_z = np.zeros(buffer_size)
        self.buffer_index = 0

        # Parámetros del filtro de Kalman
        self.estimate_x = 0
        self.estimate_y = 0
        self.estimate_z = 0
        self.error_estimate = 1  # Incertidumbre inicial
        self.error_measurement = 1  # Varianza de la medición
        self.kalman_gain = 0.5  # Ganancia inicial


    def setInitialConditions(self, initialData):
        self.x = initialData.get('x', 0)
        self.y = initialData.get('y', 0)
        self.yaw = initialData.get('yaw', 0)

    def getGpsData(self, imuData, delta_time):
        """Integración doble con ajuste de yaw."""
        imuData = ast.literal_eval(imuData)  # Convierte solo si es válido
        roll = float(imuData['roll'])
        pitch = float(imuData['pitch'])
        yaw = float(imuData['yaw'])
        accelX = float(imuData['accelx'])
        accelY = float(imuData['accely'])
        accelZ = float(imuData['accelz'])

        # Compensación de gravedad
        gravity_comp_x = np.sin(pitch) * 9.81
        gravity_comp_y = -np.sin(roll) * 9.81
        gravity_comp_z = np.cos(pitch) * np.cos(roll) * 9.81
        
        # Aplicar filtro de Kalman a cada eje
        self.estimate_x, self.error_estimate, self.kalman_gain = self.kalman_filter(
            accelX - gravity_comp_x, self.estimate_x, self.error_estimate, self.error_measurement, self.kalman_gain
        )
        self.estimate_y, self.error_estimate, self.kalman_gain = self.kalman_filter(
            accelY - gravity_comp_y, self.estimate_y, self.error_estimate, self.error_measurement, self.kalman_gain
        )
        self.estimate_z, self.error_estimate, self.kalman_gain = self.kalman_filter(
            accelZ - gravity_comp_z, self.estimate_z, self.error_estimate, self.error_measurement, self.kalman_gain
        )
        
        # Ajustar las aceleraciones con el ángulo de yaw
        adjusted_accelX = accelX * np.cos(yaw) - accelY * np.sin(yaw)
        adjusted_accelY = accelX * np.sin(yaw) + accelY * np.cos(yaw)
        # adjusted_accelZ = accelZ - gravity

        # Insertar datos en el buffer de manera circular
        self.accel_buffer_x[self.buffer_index % self.buffer_size] = adjusted_accelX
        self.accel_buffer_y[self.buffer_index % self.buffer_size] = adjusted_accelY
        # self.accel_buffer_z[self.buffer_index % self.buffer_size] = adjusted_accelZ  # No se ajusta el eje Z con yaw
        self.buffer_index += 1

        # Si el buffer está lleno, realizar integración doble
        if self.buffer_index >= self.buffer_size:
            # Primer integración: velocidad
            vel_x = np.cumsum(self.accel_buffer_x) * delta_time
            vel_y = np.cumsum(self.accel_buffer_y) * delta_time
            # vel_z = np.cumsum(self.accel_buffer_z) * delta_time

            # # Segunda integración: posición
            pos_x = np.cumsum(vel_x) * delta_time
            pos_y = np.cumsum(vel_y) * delta_time
            # pos_z = np.cumsum(vel_z) * delta_time

            # Actualizar posición con el último valor
            self.x -= pos_y[-1]/self.buffer_size
            self.y += pos_x[-1]/self.buffer_size
            # self.z += pos_z[-1]/self.buffer_size
        
        return {
            "x": self.x,
            "y": self.y,
            # "z": self.z
        }

    def kalman_filter(self, measurement, estimate, error_estimate, error_measurement, kalman_gain):
        # Predicción
        estimate = estimate  # Asumimos que no hay cambio en el estado previo

        # Actualización
        kalman_gain = error_estimate / (error_estimate + error_measurement)
        estimate = estimate + kalman_gain * (measurement - estimate)
        error_estimate = (1 - kalman_gain) * error_estimate

        return estimate, error_estimate, kalman_gain
