#!/usr/bin/env python3
import rospy
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx

# Parameters
WB = 0.26  # [m] wheel base of vehicle
target_speed = 30 / 100  # [m/s] / 100[m/cm] = [cm/s]

k = 1  # look forward gain
Lfc = 0.4  # [m] look-ahead distance
dt = 0.001  # [s] time tick

# PID parameters
Ki = 0.1  # Integral gain
Kd = 0.01  # Derivative gain
Kp = 10.0  # Proportional gain

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, ai, imu_data):
        self.x = imu_data[0]
        self.y = imu_data[1]
        self.yaw = imu_data[2]
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                if ind + 1 >= len(self.cx):
                    break
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf

class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)

class ControlSystem:
    
    def __init__(self, mode='path_tracking', sim_controller=None):
        self.sim_controller = sim_controller
        self.mode = mode
        self.integral = 0.0
        self.previous_error = 0.0
        self.state = False

    def run(self, imuData, speed):
        self._set_track("src/decision/node_following/track_test_graph.graphml")
        global target_speed

        # Trayectoria
        cx = self.cx
        cy = self.cy 

        # Con IMU
        x = imuData[0]
        y = imuData[1]
        yaw = imuData[2]

        state = State(x, y, yaw, speed)

        lastIndex = len(cx) - 1
        time = 0.0
        states = States()
        states.append(time, state)
        target_course = TargetCourse(cx, cy)
        target_ind, _ = target_course.search_target_index(state)
        self.state = True

        while lastIndex > target_ind:

            # Increment target speed at halfway point
            if target_ind >= lastIndex // 2:
                target_speed = 30 / 100  # Increase target speed to 60 cm/s

            # Calc control input
            ai, self.integral, self.previous_error = self.pid_control(target_speed, state.v, self.integral, self.previous_error)
            di, target_ind = self.pure_pursuit_steer_control(state, target_course, target_ind)

            # gps_data = self.get_gps()
            # state.update_from_gps(ai, gps_data)  
            
            
            state.update(ai,imuData)
            
            time += dt
            states.append(time, state)

            # state.v = state.v + ai * dt

            rospy.sleep(dt)
            return str(math.degrees(di)), str(state.v)
        
        print("Termino")

    def pure_pursuit_steer_control(self, state, trajectory, pind):
        ind, Lf = trajectory.search_target_index(state)

        if pind >= ind:
            ind = pind

        if ind < len(trajectory.cx):
            tx = trajectory.cx[ind]
            ty = trajectory.cy[ind]
        else:  # toward goal
            tx = trajectory.cx[-1]
            ty = trajectory.cy[-1]
            ind = len(trajectory.cx) - 1
        
        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
        alpha = -alpha
        delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0) 

        return delta, ind
    
    def pid_control(self, target, current, integral, previous_error):
        error = target - current
        integral += error * dt
        derivative = (error - previous_error) / dt
        a = Kp * error + Ki * integral + Kd * derivative
        return a, integral, error
        
    def get_state(self):
        return self.state

    def _set_track(self, path_file):
        """
        Loads the track graph from a .graphml file, calculates the global range of coordinates, 
        and sets the nodes to follow for the vehicle's path.

        Args:
            path_file (str): Path to the .graphml file containing the track graph.
        """

        # Constants for coordinate mapping
        MAPPED_X_RANGE_MIN = 0  # Minimum x value after mapping
        MAPPED_X_RANGE_MAX = 15  # Maximum x value after mapping
        MAX_X_OVERRIDE = 21  # Override value for maximum x-coordinate if applicable

        try:
            # Load the graph from the specified file
            self.track_graph = nx.read_graphml(path_file)
            print(f"Graph successfully loaded from {path_file}.")
        except Exception as e:
            print(f"Error loading the graph: {e}")
            return

        # Calculate the global range of node coordinates
        try:
            # Extract all x-coordinates from the graph nodes
            all_x = [
                float(data["x"])
                for _, data in self.track_graph.nodes(data=True)
                if "x" in data
            ]
            # Extract all y-coordinates from the graph nodes
            all_y = [
                float(data["y"])
                for _, data in self.track_graph.nodes(data=True)
                if "y" in data
            ]

            # Determine the maximum and minimum x-coordinates
            max_x, min_x = max(all_x), min(all_x)
            # Override the maximum x-coordinate value
            max_x = MAX_X_OVERRIDE  
            # Determine the maximum and minimum y-coordinates
            max_y, min_y = 15, min(all_y)

            # Print the global range of all node coordinates
            print(
                f"Global range of all nodes: x -> [min: {min_x}, max: {max_x}], "
                f"y -> [min: {min_y}, max: {max_y}]"
            )

        
        except Exception as e:
            print(f"Error calculating the global coordinate range: {e}")
            return

        '''# Request the user to input the nodes for the path
        node_input = input(
            "Enter the nodes to follow, separated by commas (e.g., Node1,Node2,Node3): "
        ).strip()
        # Handle empty input
        if not node_input:
            print("Error: No nodes were entered.")
            return'''
        
        # Define the input nodes as a string
        node_input = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33"

        # Create a list of nodes from the input string, removing any extra spaces
        self.path = [node.strip() for node in node_input.split(",")]

        # Validate that the nodes exist in the graph
        invalid_nodes = [node for node in self.path if node not in self.track_graph.nodes]
        
        # If there are any invalid nodes, print an error message and return
        if invalid_nodes:
            print(f"Error: The following nodes do not exist in the graph: {', '.join(invalid_nodes)}")
            return

        # Print the valid path nodes
        print(f"Path set: {self.path}")
        # Extract and map the coordinates of the path nodes
        try:
            # Extract the original coordinates of the path nodes from the graph
            original_x = [float(self.track_graph.nodes[node]["x"]) for node in self.path]
            original_y = [float(self.track_graph.nodes[node]["y"]) for node in self.path]

            self.cx = original_x
            self.cy = original_y  

            # Print a success message indicating that the trajectory coordinates were calculated successfully
            print("Trajectory coordinates calculated successfully.")
            

        except KeyError as e:
            print(f"Error: Missing coordinate '{e}' in one of the nodes.")
        except Exception as e:
            print(f"Error calculating the coordinates: {e}")
