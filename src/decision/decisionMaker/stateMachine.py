import time
import networkx as nx
import numpy as np # grafo

from src.decision.lineFollowing.purepursuitpd import Controller
from src.decision.distance.distanceModule import DistanceModule
from src.decision.lineFollowing.purepursuit import ControlSystem
from src.utils.constants import (
    #States
    start_state, end_state, lane_following, classifying_sign, stop_state, parking_state,
    waiting_for_pedestrian, overtaking_moving_car, overtaking_static_car, avoiding_roadblock,
    classifying_obstacle, tailing_car, approaching_stop_line, crosswalk_navigation,
    roundabout_navigation, tracking_local_path, intersection_navigation, waiting_at_stopline,
    waiting_for_green, highway_state, crosswalk_state,priority_state, traffic_light_state,
    red_state, yellow_state, stopline_state, pedestrian_crossing, continue_line,

    #Events
    ROADMAP_LOADED, OBSTACLE_DISTANCE_THRESHOLD, SIGN_DISTANCE_THRESHOLD, END_EVENT,
    STOP_LINE_APPROACH_DISTANCE_THRESHOLD, PARKING_SIGN_DETECTED, TRY_PARKING, CONTINUE_LANE_FOLLOWING,
    STOP_SIGN_DETECTED, TIMEOUT_STOP, PARKING_COMPLETED, PEDESTRIAN_TIMEOUT, CAR_OVERTAKEN,
    ROADBLOCK_AVOIDED, OBSTACLE_PEDESTRIAN, OBSTACLE_CAR, OBSTACLE_ROADBLOCK, IF_OBSTACLE_TOO_FAR,
    IF_MOVING, IF_STATIC, CROSSWALK_SIGN_DETECTED, TIMEOUT_CROSSWALK, INTERSECTION_TRAFFIC_LIGHT_EVENT,
    INTERSECTION_STOP_EVENT, JUNCTION_EVENT, INTERSECTION_PRIORITY_EVENT, ROUNDABOUT_EVENT,
    ALWAYS, END_OF_LOCAL_PATH, TIMEOUT_STOPLINE, SEMAPHORE_GREEN, WAITING_STOP,HIGHWAY_ENTRY_SIGN_DETECTED,
    HIGHWAY_END_SIGN_DETECTED, IN_HIGHWAY, ONE_WAY_SIGN_DETECTED, ROUNDABOUT_SIGN_DETECTED,
    NO_ENTRY_SIGN_DETECTED, PEDESTRIAN_DETECTED, CAR_DETECTED, ROADBLOCK_DETECTED,
    TRACKING_NODE, FINAL_NODE_REACHED, WAITING_CROSSWALK, CROSSWALK_TIMEOUT, PRIORITY_INTERSECTION,
    RED_LIGHT_DETECTED, YELLOW_LIGHT_DETECTED, WAITING_RED_LIGHT, RED_LIGHT_FINISHED, WAITING_YELLOW_LIGHT, YELLOW_LIGHT_FINISHED,
    GREEN_LIGHT_DETECTED, WAITING_STOPLINE, WAITING_PEDESTRIAN, PEDESTRIAN_CROSSED, TRY_AVOID_ROADBLOCK,
    IF_CONTINUE_LINE, WAITING_OBSTACLE, OBSTACLE_GONE,
    # Signs
    STOP_SIGN, PARKING_SIGN, CROSSWALK_SIGN, PRIORITY_SIGN, HIGHWAY_ENTRY_SIGN, PRIORITY_SIGN_DETECTED,
    HIGHWAY_END_SIGN, ONE_WAY_SIGN, ROUNDABOUT_SIGN, NO_ENTRY_SIGN,

    # Obstacles
    PEDESTRIAN, CAR, ROADBLOCK,

    # Rutinas
    control_for_signs
)

ALWAYS_ON_ROUTINES = [control_for_signs]
LANE_FRAME_SKIP = 1
SIGNAL_FRAME_SKIP = 25


class Routine():
    def __init__(self, name, method, activated=False):
        self.name = name
        self.method = method
        self.active = activated

    def __str__(self):
        return self.name

    def run(self):
        self.method()


class StateMachine():
    def __init__(self):
        self.distance_module = DistanceModule()
        self.control_system = Controller()
        self.desired_speed = 0.1
        
        self.state_transitions = {
            start_state: {ROADMAP_LOADED: lane_following},
            lane_following: {
                OBSTACLE_DISTANCE_THRESHOLD: classifying_obstacle,
                STOP_LINE_APPROACH_DISTANCE_THRESHOLD: approaching_stop_line,
                SIGN_DISTANCE_THRESHOLD: classifying_sign,
                END_EVENT: end_state,
                CONTINUE_LANE_FOLLOWING: lane_following,
            },
            classifying_sign: {
                PRIORITY_SIGN_DETECTED: priority_state,
                STOP_SIGN_DETECTED: stop_state,
                PARKING_SIGN_DETECTED: parking_state,
                HIGHWAY_ENTRY_SIGN_DETECTED: highway_state,
                CROSSWALK_SIGN_DETECTED: crosswalk_state,
                ONE_WAY_SIGN_DETECTED: tracking_local_path,
                ROUNDABOUT_SIGN_DETECTED: tracking_local_path,
                NO_ENTRY_SIGN_DETECTED: tracking_local_path,
            },
            priority_state: {
                PRIORITY_INTERSECTION: approaching_stop_line,
            },
            stop_state: {
                WAITING_STOP: stop_state,
                TIMEOUT_STOP: lane_following
            },
            parking_state: {
                PARKING_COMPLETED: lane_following, 
                TRY_PARKING: parking_state
            },
            highway_state: {
                IN_HIGHWAY: highway_state,
                HIGHWAY_END_SIGN_DETECTED: lane_following,
            },
            crosswalk_state: {
                WAITING_CROSSWALK: crosswalk_state,
                CROSSWALK_TIMEOUT: lane_following, 
            },
            traffic_light_state: {
                RED_LIGHT_DETECTED: red_state,
                YELLOW_LIGHT_DETECTED: yellow_state,
                GREEN_LIGHT_DETECTED: lane_following
            },
            red_state: {
                WAITING_RED_LIGHT: red_state,
                RED_LIGHT_FINISHED: traffic_light_state,
            },
            yellow_state: {
                WAITING_YELLOW_LIGHT: yellow_state,
                YELLOW_LIGHT_FINISHED: traffic_light_state,
            },
            tracking_local_path: {
                TRACKING_NODE: tracking_local_path,
                FINAL_NODE_REACHED: lane_following,
            },
            approaching_stop_line: {
                INTERSECTION_STOP_EVENT: stopline_state,
                INTERSECTION_PRIORITY_EVENT: tracking_local_path,
            },
            stopline_state: {
                WAITING_STOPLINE: stopline_state,
                TIMEOUT_STOPLINE: tracking_local_path,
            },
            classifying_obstacle: {
                PEDESTRIAN_DETECTED: pedestrian_crossing,
                ROADBLOCK_DETECTED: avoiding_roadblock,
                CAR_DETECTED: tailing_car,
            },
            pedestrian_crossing: {
                WAITING_PEDESTRIAN: pedestrian_crossing,
                PEDESTRIAN_CROSSED: lane_following,
            },
            avoiding_roadblock: {
                TRY_AVOID_ROADBLOCK: avoiding_roadblock,
                PEDESTRIAN_CROSSED: lane_following,
                IF_CONTINUE_LINE: continue_line,
            },
            continue_line: {
                WAITING_OBSTACLE: continue_line,
                OBSTACLE_GONE: lane_following,
            },
            tailing_car: {
                IF_CONTINUE_LINE: continue_line,
                IF_STATIC: overtaking_static_car,
                IF_MOVING: overtaking_moving_car,
            },
            overtaking_static_car: {
                CAR_OVERTAKEN: lane_following
            },
            overtaking_moving_car: {
                CAR_OVERTAKEN: lane_following
            },
        }
        
        self.event_methods = {
            ROADMAP_LOADED: self.on_roadmap_loaded,
            OBSTACLE_DISTANCE_THRESHOLD: self.on_obstacle_detected,
            END_EVENT: self.on_end,
            STOP_LINE_APPROACH_DISTANCE_THRESHOLD: self.on_stop_line_approach,
            CONTINUE_LANE_FOLLOWING: self.on_lane_following,
            PEDESTRIAN_TIMEOUT: self.on_pedestrian_timeout,
            CAR_OVERTAKEN: self.on_car_overtaken,
            ROADBLOCK_AVOIDED: self.on_roadblock_avoided,
            OBSTACLE_PEDESTRIAN: self.on_pedestrian_detected,
            OBSTACLE_CAR: self.on_car_detected,
            OBSTACLE_ROADBLOCK: self.on_roadblock_detected,
            IF_OBSTACLE_TOO_FAR: self.on_obstacle_too_far,
            IF_MOVING: self.on_car_moving,
            IF_STATIC: self.on_car_static,
            CROSSWALK_SIGN_DETECTED: self.on_crosswalk_sign_detected,
            TIMEOUT_CROSSWALK: self.on_crosswalk_timeout,
            INTERSECTION_TRAFFIC_LIGHT_EVENT: self.on_waiting_for_green,
            INTERSECTION_STOP_EVENT: self.on_waiting_at_stopline,
            SEMAPHORE_GREEN: self.on_semaphore_green,
            TIMEOUT_STOPLINE: self.on_timeout_stopline,

            PARKING_SIGN_DETECTED: self.on_parking,
            TRY_PARKING: self.try_parking,
            PARKING_COMPLETED: self.on_parking_completed,

            STOP_SIGN_DETECTED: self.on_stop_sign_detected,
            WAITING_STOP: self.on_waiting_stop,
            TIMEOUT_STOP: self.on_timeout_stop,


            HIGHWAY_ENTRY_SIGN_DETECTED: self.on_highway_entry_sign_detected,
            IN_HIGHWAY: self.on_highway,
            HIGHWAY_END_SIGN_DETECTED: self.on_highway_end_sign_detected,
        
            ONE_WAY_SIGN_DETECTED: self.on_one_way_sign_detected,
            ROUNDABOUT_SIGN_DETECTED: self.on_roundabout_way_sign_detected,
            NO_ENTRY_SIGN_DETECTED: self.on_no_entry_sign_detected,
        }

        
        # PARKING
        self.parking_start_time = time.time()  
        self.parking_step = 0  
        self.parking_duration = [3, 1, 2, 2, 2, 3]  
        self.parking_end_time = None

        # HIGHWAY
        self.highway_end = False  
        
        # STOP
        self.stop_time = 3
        # CROSSWALK
        self.timeout_crosswalk = 6
        self.in_crosswalk = False

        self.current_state = lane_following #start_state

        self.current_speed = None
        self.current_steer = None
        self.current_direction = None
        self.current_deviation = None
        self.objects_detected = None
        self.current_ultra_values = None

    #===================== STATE HANDLING =====================#
    def change_state(self, event):
        """Cambia el estado basándose en el evento."""
        if self.current_state in self.state_transitions and event in self.state_transitions[self.current_state]:
            new_state = self.state_transitions[self.current_state][event]
            #print(f"Cambiando de estado: {self.current_state} -> {new_state} por evento: {event}")
            self.event_methods[event]()
            self.current_state = new_state
        else:
            print(f"No se puede cambiar al estado con el evento: {event} desde {self.current_state}")
    #===================== EVENT HANDLING =====================#
    def handle_events(self, act_deviation, objects_detected, current_speed, current_steer, direction, ultra_values):
        self.current_deviation = act_deviation if act_deviation is not None else self.current_deviation
        self.objects_detected = objects_detected if objects_detected is not None else self.objects_detected
        self.current_speed = current_speed if current_speed is not None else self.current_speed
        self.current_steer = current_steer if current_steer is not None else self.current_steer
        self.current_direction = direction if direction is not None else self.current_direction
        self.current_ultra_values = ultra_values if ultra_values is not None else self.current_ultra_values
        
        if self.current_state == lane_following:
            if objects_detected and any(valid_distance for _, valid_distance in objects_detected):
                self.change_state(SIGN_DISTANCE_THRESHOLD)

            self.change_state(CONTINUE_LANE_FOLLOWING)
        
        elif self.current_state == classifying_sign and objects_detected:
            for sign_name, valid_distance in objects_detected:
                if sign_name == STOP_SIGN and valid_distance:
                    self.change_state(STOP_SIGN_DETECTED)
                elif sign_name == PARKING_SIGN and valid_distance:
                    self.change_state(PARKING_SIGN_DETECTED)
                elif sign_name == CROSSWALK_SIGN and valid_distance:
                    self.change_state(CROSSWALK_SIGN_DETECTED)
                elif sign_name == PRIORITY_SIGN and valid_distance:
                    #TODO: talvez desactivar la deteccion de autos con una variable sign_name == "CAR" and valid_distance and priority == False
                    # pero podria generar problemas al tener un auto de frente y con velocidad menor a la nuestra
                    self.change_state(PRIORITY_SIGN_DETECTED)
                elif sign_name == HIGHWAY_ENTRY_SIGN and valid_distance:
                    self.change_state(PARKING_SIGN_DETECTED)
                elif sign_name == ONE_WAY_SIGN and valid_distance:
                    #TODO: desactivar el intento de traspasar autos creo
                    self.change_state(ONE_WAY_SIGN_DETECTED)
                elif sign_name == ROUNDABOUT_SIGN and valid_distance:
                    #TODO: activar seguimiento de nodos para movernos
                    self.change_state(ROUNDABOUT_SIGN_DETECTED)
                elif sign_name == NO_ENTRY_SIGN and valid_distance:
                    #TODO: tachar un nodo de la lista de nodos como no utilizable
                    self.change_state(NO_ENTRY_SIGN_DETECTED)
                elif sign_name == PEDESTRIAN and valid_distance:
                    #TODO: parar hasta que deje de detectar
                    self.change_state(PEDESTRIAN_DETECTED)
                elif sign_name == CAR and valid_distance:
                    #TODO: parar hasta que deje de detectar
                    self.change_state(CAR_DETECTED)
                elif sign_name == ROADBLOCK and valid_distance:
                    #TODO: esquivar
                    self.change_state(ROADBLOCK_DETECTED)

        elif self.current_state == stop_state:
            if self.timeout_stop == -1:
                self.change_state(TIMEOUT_STOP) 
            else:
                self.change_state(WAITING_STOP)
        elif self.current_state == parking_state:
            if self.parking_end_time == -1:
                self.change_state(PARKING_COMPLETED)
            else:
                self.change_state(TRY_PARKING)
        elif self.current_state == highway_state:
            if self.highway_end == True:
                self.change_state(HIGHWAY_END_SIGN_DETECTED)
            else:
                self.change_state(IN_HIGHWAY)
        
        elif self.current_state == classifying_obstacle and objects_detected:
            for object_name, valid_distance in objects_detected:
                if object_name == PEDESTRIAN and valid_distance:
                    self.change_state(OBSTACLE_PEDESTRIAN)
                elif object_name == CAR and valid_distance:
                    self.change_state(OBSTACLE_CAR)
                elif object_name == ROADBLOCK and valid_distance:
                    self.change_state(OBSTACLE_ROADBLOCK)
        return self.current_speed, self.current_steer
     
    def on_roadmap_loaded(self): 
        path_file = "src/example/src/nodes_following/Competition_track_graph.graphml"

        self.track_graph = nx.read_graphml(path_file)
        # esto va para tracking_local_path, self.path serian los nodos a tomar
        #self.cx = [float(self.track_graph.nodes[node]["x"]) for node in self.path]
        #self.cy = [float(self.track_graph.nodes[node]["y"]) for node in self.path]
        # roundabout_event setea path y TRACKING_NODE ejecuta pure pursuit

    def on_obstacle_detected(self): pass

    def on_end(self): 
        self.current_speed = 0
        self.current_steer = 0

    def on_stop_line_approach(self): pass

    def on_parking(self):
        self.parking_start_time = time.time()  
        self.parking_step = 0 
        self.a=False
        self.b = False
        self.c = False
        self.d = False
    def try_parking(self): 
        current_time = time.time()  
        elapsed_time = current_time - self.parking_start_time 

        if self.parking_step == 0:  
            self.current_speed = "200"
            self.current_steer = "0"
            if elapsed_time >= self.parking_duration[0]:
                self.parking_step += 1
                self.parking_start_time = current_time  
        elif self.parking_step == 1:  
            self.current_speed = "0"
            self.current_steer = "240"
            if elapsed_time >= self.parking_duration[1]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 2:  
            self.current_speed = "-100"
            self.current_steer = "240"
            if elapsed_time >= self.parking_duration[2]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 3: 
            self.current_speed = "-100"
            self.current_steer = "0"
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 4: 
            self.current_speed = "-100"
            self.current_steer = "0"
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 5: 
            self.current_speed = "0"
            self.current_steer = "-240"
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 6: 
            self.current_speed = "-100"
            self.current_steer = "-240"
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 7:
            self.current_speed = "0"
            self.current_steer = "0"
            if elapsed_time >= self.parking_duration[4]:
                self.parking_step = 0  
                self.parking_start_time = None 
                self.parking_end_time = current_time  
                self.on_parking_completed()  

    def on_parking_completed(self): 
        print("Secuencia de estacionamiento completada.")
        self.parking_end_time = -1
        #end time setear en None y en handle events consultar esta variable

    def on_lane_following(self): 
        speed, angle_ref = self.control_system.get_control(self.current_deviation, self.current_direction, 0, self.desired_speed)
        angle_ref = np.rad2deg(angle_ref)
        self.current_steer = f"{int((angle_ref + 3 )* 10)}"
        self.current_speed = f"{int(speed * 1000)}"


    def on_stop_sign_detected(self):
        # TODO: revisar valor de stop que recibe el modulo
        self.timeout_stop = time.time()
        self.current_speed = self.distance_module.handle_stop_signal_logic(self.objects_detected["STOP"], self.current_speed)
    def on_waiting_stop(self):
        current_time = time.time()  
        elapsed_time = current_time - self.timeout_stop 
        if elapsed_time >= self.stop_time:
            self.on_timeout_stop()
    def on_timeout_stop(self):
        self.timeout_stop = -1 
        
    def on_highway_entry_sign_detected(self):
        self.highway_end = False
        self.current_speed(300)
    def on_highway(self):
        # TODO:revisa posicion actual, nodo actual y ve si cambia o no. al llegar al nodo final, levanta una bandera aunque ya deberia
        # haber detectado la senal de fin de highway POSIBLENTE NO SEA NECESARIO CONSULTAR NODOS
        # si da igual el tema de nodos, entonce esta funcion no sirve, cuando detecta, en on highway entry sign detected, aumenta velocidad, guardando la actual, y cuando detecta el fin, retoma la velocidad inicla
        if any(sign == HIGHWAY_END_SIGN and valid_distance for sign, valid_distance in self.objects_detected):
                self.on_highway_end_sign_detected()
        pass
    def on_highway_end_sign_detected(self):
        self.highway_end = True

    def on_one_way_sign_detected(self):
        final_node_reached = False

        #TODO: inicializar nodo inicial, nodo a seguir y nodo final del path local tracking
        pass
    def on_roundabout_way_sign_detected(self):
        final_node_reached = False

        #TODO: inicializar nodo inicial, nodo a seguir y nodo final del path local tracking
        pass
    def on_no_entry_sign_detected(self):
        #TODO: inicializar nodo inicial, nodo a seguir y nodo final del path local tracking
        final_node_reached = False
        pass
    def on_tracking_node(self):
        # if actual_node == final_node: self.on_final_node_reached()
        # TODO: aca deberia setear el siguiente nodo a seguir modificando direccion
        pass
    def on_final_node_reached(self):
        final_node_reached = True

    def on_crosswalk_sign_detected(self):
        self.crosswalk_speed = self.current_speed
        self.timeout_crosswalk = time.time()
        self.current_speed(50)
        
        # SOL A
        self.in_crosswalk = True
    def on_crosswalk_waiting(self):
        current_time = time.time()  
        elapsed_time = current_time - self.timeout_crosswalk 
        if elapsed_time >= self.stop_time:
            self.on_crosswalk_timeout()
    def on_crosswalk_timeout(self):
        self.current_speed = self.crosswalk_speed


        # SOL A
        self.in_crosswalk = False

    def on_pedestrian_timeout(self): pass
    def on_car_overtaken(self): pass
    def on_roadblock_avoided(self): pass
    def on_pedestrian_detected(self): pass
    def on_car_detected(self): pass
    def on_roadblock_detected(self): pass
    def on_obstacle_too_far(self): pass
    def on_car_moving(self): pass
    def on_car_static(self): pass
    def on_waiting_for_green(self): pass
    def on_waiting_at_stopline(self): pass
    def on_timeout_stopline(self): pass
    def on_semaphore_green(self): pass
'''
    #===================== STATE MACHINE MANAGEMENT =====================#
    def run(self):
        """Método principal que se ejecuta en cada iteración del ciclo."""
        self.run_current_state()
        self.run_active_routines()
        self.log_status()

    def run_current_state(self):
        state_method = self.state_methods.get(self.current_state)
        state_method() 

    def run_active_routines(self):
        """Ejecuta todas las rutinas activas y las que están en ALWAYS_ON_ROUTINES."""
        for routine in self.routines.values():
            if routine.active:
                routine.run()
        for routine_name in ALWAYS_ON_ROUTINES:
            if routine_name in self.routines:
                self.routines[routine_name].run()
            else:
                print(f"WARNING: Rutina '{routine_name}' no está definida en self.routines.")

    def log_status(self):
        """Muestra el estado actual del sistema."""
        print(f'CURR_SIGN: {self.curr_sign}')
        print(f'CURR_LINES: {self.act_lines}')
        print(f'CURR_DEV: {self.act_deviation}')
'''