import numpy as np

class imu_gps():
    def __init__(self, buffer_size=100):
        self.x = 0
        self.y = 0
        self.z = 0

        self.buffer_size = buffer_size
        self.accel_buffer_x = np.zeros(buffer_size)
        self.accel_buffer_y = np.zeros(buffer_size)
        self.accel_buffer_z = np.zeros(buffer_size)
        self.buffer_index = 0

    def setInitialConditions(self, initialData):
        self.x = initialData.get('x', 0)
        self.y = initialData.get('y', 0)
        self.yaw = initialData.get('yaw', 0)

    def getGpsData(self, imuData, delta_time):
        """Integración doble con ajuste de yaw."""
        accelX = getattr(imuData, "accelx", 0)
        accelY = getattr(imuData, "accely", 0)
        accelZ = getattr(imuData, "accelz", 0)
        yaw = getattr(imuData, "yaw", 0)

        # Ajustar las aceleraciones con el ángulo de yaw
        adjusted_accelX = accelX * np.cos(yaw) - accelY * np.sin(yaw)
        adjusted_accelY = accelX * np.sin(yaw) + accelY * np.cos(yaw)

        # Insertar datos en el buffer de manera circular
        self.accel_buffer_x[self.buffer_index % self.buffer_size] = adjusted_accelX
        self.accel_buffer_y[self.buffer_index % self.buffer_size] = adjusted_accelY
        self.accel_buffer_z[self.buffer_index % self.buffer_size] = accelZ  # No se ajusta el eje Z con yaw
        self.buffer_index += 1

        # Si el buffer está lleno, realizar integración doble
        if self.buffer_index >= self.buffer_size:
            # Primer integración: velocidad
            vel_x = np.cumsum(self.accel_buffer_x) * delta_time
            vel_y = np.cumsum(self.accel_buffer_y) * delta_time
            vel_z = np.cumsum(self.accel_buffer_z) * delta_time

            # # Segunda integración: posición
            pos_x = np.cumsum(vel_x) * delta_time
            pos_y = np.cumsum(vel_y) * delta_time
            pos_z = np.cumsum(vel_z) * delta_time

            # Actualizar posición con el último valor
            self.x -= pos_y[-1]/self.buffer_size
            self.y += pos_x[-1]/self.buffer_size
            self.z += pos_z[-1]/self.buffer_size

        return {
            "x": self.x,
            "y": self.y,
            "z": self.z
        }

