#!/usr/bin/env python3
import numpy as np
import math

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

# class State:

#     def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
#         self.x = x
#         self.y = y
#         self.yaw = yaw
#         self.v = v
#         self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
#         self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

#     def update(self, delta):
#         self.x += self.v * math.cos(self.yaw) * dt
#         self.y += self.v * math.sin(self.yaw) * dt
#         self.yaw += self.v / WB * math.tan(delta) * dt
#         self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
#         self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

#     def update_from_gps(self, a, gps_data):
#         """
#         Actualiza el estado utilizando datos de GPS.
        
#         Args:
#             gps_data (dict): Diccionario con datos de GPS, como 'x', 'y', 'yaw', y 'v'.
#         """
#         self.x = gps_data.posA
#         self.y = gps_data.posB 
#         self.yaw = (gps_data.rotA) % (2 * math.pi)  # Restamos pi/2 para ajustar la orientación
#         self.v += a * dt
#         self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
#         self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))


#     def calc_distance(self, point_x, point_y):
#         dx = self.rear_x - point_x
#         dy = self.rear_y - point_y
#         return math.hypot(dx, dy)

# class TargetCourse:

#     def __init__(self, cx, cy):
#         self.cx = cx
#         self.cy = cy
#         self.old_nearest_point_index = None

#     def search_target_index(self, state):

#         # To speed up nearest point search, doing it at only first time.
#         if self.old_nearest_point_index is None:
#             # search nearest point index
#             dx = [state.rear_x - icx for icx in self.cx]
#             dy = [state.rear_y - icy for icy in self.cy]
#             d = np.hypot(dx, dy)
#             ind = np.argmin(d)
#             self.old_nearest_point_index = ind
#         else:
#             ind = self.old_nearest_point_index
#             distance_this_index = state.calc_distance(self.cx[ind],
#                                                       self.cy[ind])
#             while True:
#                 if ind + 1 >= len(self.cx):
#                     break
#                 distance_next_index = state.calc_distance(self.cx[ind + 1],
#                                                           self.cy[ind + 1])
#                 if distance_this_index < distance_next_index:
#                     break
                
#                 ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                
#                 distance_this_index = distance_next_index
#             self.old_nearest_point_index = ind

#         Lf = k * state.v + Lfc  # update look ahead distance

#         # search look ahead target point index
#         while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
#             if (ind + 1) >= len(self.cx):
#                 break  # not exceed goal
#             ind += 1

#         return ind, Lf

# class States:

#     def __init__(self):
#         self.x = []
#         self.y = []
#         self.yaw = []
#         self.v = []
#         self.t = []

#     def append(self, t, state):
#         self.x.append(state.x)
#         self.y.append(state.y)
#         self.yaw.append(state.yaw)
#         self.v.append(state.v)
#         self.t.append(t)

# def pure_pursuit_steer_control(state, trajectory, pind):
#     ind, Lf = trajectory.search_target_index(state)

#     if pind >= ind:
#         ind = pind

#     if ind < len(trajectory.cx):
#         tx = trajectory.cx[ind]
#         ty = trajectory.cy[ind]
#     else:  # toward goal
#         tx = trajectory.cx[-1]
#         ty = trajectory.cy[-1]
#         ind = len(trajectory.cx) - 1
    
#     alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
#     alpha = -alpha
#     delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0) 

#     return delta, ind

# def pid_control(target, current, integral, previous_error):
#     error = target - current
#     integral += error * dt
#     derivative = (error - previous_error) / dt
#     a = Kp * error + Ki * integral + Kd * derivative
#     return a, integral, error


class ControlSystem:
    def __init__(self, mode='path_tracking'):
        self.mode = mode
        self.previous_error = 0.0


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
