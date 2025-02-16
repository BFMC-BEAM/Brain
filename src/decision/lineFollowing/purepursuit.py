#!/usr/bin/env python3
import numpy as np
import math
import time

# Parameters
WB = 0.26  # [m] wheel base of vehicle
target_speed = 30 / 100  # [m/s] / 100[m/cm] = [cm/s]

k = 1  # look forward gain
Lfc = .5  # [m] look-ahead distance
dt = 0.001  # [s] time tick

# PID parameters
Ki = 0.1  # Integral gain
Kd = 0.01  # Derivative gain
Kp = 10.0  # Proportional gain

class ControlSystem:
    def __init__(self, mode='path_tracking'):
        self.mode = mode
        self.previous_error = 0.0
        self.prev_time = time.time()  # Inicializar el tiempo anterior
        self.integral = 0.0  # Inicializar el término integral


    def adjust_direction(self, deviation, direction):
        """
        Adjusts the steering angle based on lane deviation and direction.

        Args:
            deviation (float): The deviation from the center of the lane.
            direction (str): The direction of deviation ("left" or "right").
        """
        # Constants for proportional and derivative gains
        KP = 1  # Proportional gain for steering control
        KD = 0.1  # Derivative gain for smoothing adjustments
        MAX_STEERING_ANGLE = 24  # Maximum allowable steering angle in degrees
        MIN_TURN_ANGLE = 1  # Minimum turn angle to ensure response
        
        if deviation is None:
            return

        # Calculate proportional and derivative terms
        error = deviation
        derivative = error - self.previous_error
        steering_angle = KP * error + KD * derivative

        # Limit the steering angle within the maximum range
        steering_angle = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, math.degrees(steering_angle)))

        # Ensure minimum turn angles when deviation is detected
        if direction == "right":
            steering_angle = max(steering_angle, MIN_TURN_ANGLE)
        elif direction == "left":
            steering_angle = min(steering_angle, -MIN_TURN_ANGLE)
        else:
            steering_angle = 0  # Keep straight if no deviation detected

        # Apply the steering angle to the simulation controllerstr(new_steer)
        self.previous_error = error

        # self.sim_controller.set_steering_angle(steering_angle)
        return int(steering_angle)

    def adjust_direction_PID(self, deviation, direction):
        """
        Ajusta el ángulo de dirección basado en la desviación usando un controlador PID.
        """
        # PID controller parameters
        KP = 2.0  # Proporcional
        KI = 0.2  # Integral
        KD = 0.1  # Derivativo
        MAX_STEERING_ANGLE = 24  # Ángulo máximo de dirección

        if deviation is None:
            return

        # Obtener el tiempo actual y calcular dt
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time  # Actualizar el tiempo anterior

        print(f"current_time: {current_time:.6f}")
        print(f"previous_time: {self.prev_time:.6f}")
        print(f"dt: {dt:.6f}")

        # Evitar división por cero en la derivada
        if dt == 0:
            dt = 1e-6

        # Calcular términos PID
        error = deviation
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        steering_angle = KP * error + KI * self.integral + KD * derivative

        # Limitar el ángulo de dirección
        steering_angle = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steering_angle))

        # Actualizar error previo
        self.previous_error = error

        return int(steering_angle)

    def pure_pursuit(self, deviation, direction):

        # Parámetros de Pure Pursuit
        LOOKAHEAD_DISTANCE = .3  # Ajuste dinámico según velocidad
        WHEELBASE = 0.26  # Distancia entre ejes del vehículo [m]
        MAX_STEERING_ANGLE = 24  # Límite máximo del ángulo de dirección


        if deviation is None:
            return

        # Definir el punto de mira (lookahead) basado en la desviación detectada
        x_lookahead = deviation # El punto está desplazado en la dirección del error
        # y_lookahead = LOOKAHEAD_DISTANCE  # Distancia fija hacia adelante

        # Cálculo de la curvatura kappa
        curvature = (2 * x_lookahead) / (LOOKAHEAD_DISTANCE ** 2)

        
        # Cálculo del ángulo de dirección usando la ecuación de Ackermann
        steering_angle_rad = math.atan(WHEELBASE * curvature)
        steering_angle_deg = math.degrees(steering_angle_rad)

        # Limitar el ángulo de dirección
        steering_angle_deg = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steering_angle_deg))
        
        # # Debugging
        # print("dev [cm]: ", deviation * 100)
        # print("Curvature: ", curvature)
        # print("Steering: ", steering_angle_deg)

        return int(steering_angle_deg)
