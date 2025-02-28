import time
import networkx as nx
import numpy as np # grafo

from src.decision.distance.distanceModule import DistanceModule
from src.decision.lineFollowing.purepursuitpd import Controller
from src.utils.constants import (
    #States
    start_state, end_state, lane_following, classifying_signal, stop_state, parking_state,
    overtaking_moving_car, overtaking_static_car, avoiding_roadblock,
    classifying_obstacle, tailing_car, approaching_stop_line,
    tracking_local_path, highway_state, crosswalk_state, traffic_light_state,
    red_state, stopline_state, pedestrian_crossing, continue_lane, yellow_state,

    #Events
    ROADMAP_LOADED, OBSTACLE_DISTANCE_THRESHOLD, SIGN_DISTANCE_THRESHOLD, END_EVENT,
    STOP_LINE_APPROACH_DISTANCE_THRESHOLD, PARKING_SIGN_DETECTED, TRY_PARKING, CONTINUE_LANE_FOLLOWING,
    STOP_SIGN_DETECTED, TIMEOUT_STOP, PARKING_COMPLETED, CAR_OVERTAKEN,
    ROADBLOCK_AVOIDED, IF_MOVING, IF_STATIC, CROSSWALK_SIGN_DETECTED, CROSSWALK_WAITING,
    STOPLINE_TIMEOUT, WAITING_STOP,HIGHWAY_ENTRY_SIGN_DETECTED,
    HIGHWAY_END_SIGN_DETECTED, IN_HIGHWAY, ONE_WAY_SIGN_DETECTED, ROUNDABOUT_SIGN_DETECTED,
    NO_ENTRY_SIGN_DETECTED, PEDESTRIAN_DETECTED, CAR_DETECTED, ROADBLOCK_DETECTED,
    TRACKING_NODE, FINAL_NODE_REACHED, WAITING_CROSSWALK, CROSSWALK_TIMEOUT, INTERSECTION_PRIORITY,
    RED_LIGHT_DETECTED, YELLOW_LIGHT_DETECTED, RED_LIGHT_WAITING, RED_LIGHT_FINISHED, YELLOW_LIGHT_WAITING, YELLOW_LIGHT_FINISHED,
    GREEN_LIGHT_DETECTED, STOPLINE_WAITING, PEDESTRIAN_WAITING, PEDESTRIAN_CROSSED, TRY_AVOID_ROADBLOCK,
    IN_CONTINUE_LANE, WAITING_OBSTACLE, OBSTACLE_GONE, INTERSECTION_PRIORITY_EVENT, INTERSECTION_STOP, ULTRA_SOUND_EMERGENCY,
    TRAFFIC_LIGHT_DETECTED, CAR_MOVING, CAR_STATIC, CONTINUE_LANE_DRIVING,
    OBSTACLE_WAITING,ROADBLOCK_AVOIDING,
    # Signs
    STOP_SIGN, PARKING_SIGN, CROSSWALK_SIGN, PRIORITY_SIGN, HIGHWAY_ENTRY_SIGN, PRIORITY_SIGN_DETECTED,
    HIGHWAY_END_SIGN, ONE_WAY_SIGN, ROUNDABOUT_SIGN, NO_ENTRY_SIGN,

    # Obstacles
    PEDESTRIAN, CAR, ROADBLOCK,

    # Traffic light
    RED, YELLOW, GREEN
)

class StateMachine():
    def __init__(self):
        self.distance_module = DistanceModule()
        self.control_system = Controller()
        self.desired_speed = 0.3

        
        self.state_transitions = {
            start_state: {ROADMAP_LOADED: lane_following},
            lane_following: {
                OBSTACLE_DISTANCE_THRESHOLD: classifying_obstacle,
                STOP_LINE_APPROACH_DISTANCE_THRESHOLD: approaching_stop_line,
                SIGN_DISTANCE_THRESHOLD: classifying_signal,
                END_EVENT: end_state,
                CONTINUE_LANE_FOLLOWING: lane_following,
                ULTRA_SOUND_EMERGENCY: classifying_obstacle,
            },
            classifying_signal: {
                PRIORITY_SIGN_DETECTED: approaching_stop_line,
                STOP_SIGN_DETECTED: stop_state,
                PARKING_SIGN_DETECTED: parking_state,
                HIGHWAY_ENTRY_SIGN_DETECTED: highway_state,
                CROSSWALK_SIGN_DETECTED: crosswalk_state,
                ONE_WAY_SIGN_DETECTED: tracking_local_path,
                ROUNDABOUT_SIGN_DETECTED: tracking_local_path,
                NO_ENTRY_SIGN_DETECTED: tracking_local_path,
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
                RED_LIGHT_WAITING: red_state,
                RED_LIGHT_FINISHED: traffic_light_state,
            },
            yellow_state: {
                YELLOW_LIGHT_WAITING: yellow_state,
                YELLOW_LIGHT_FINISHED: traffic_light_state,
            },
            tracking_local_path: {
                TRACKING_NODE: tracking_local_path,
                FINAL_NODE_REACHED: lane_following,
            },
            approaching_stop_line: {
                INTERSECTION_STOP: stopline_state,
                INTERSECTION_PRIORITY_EVENT: tracking_local_path,
            },
            stopline_state: {
                STOPLINE_WAITING: stopline_state,
                STOPLINE_TIMEOUT: lane_following,
            },
            classifying_obstacle: {
                PEDESTRIAN_DETECTED: pedestrian_crossing,
                ROADBLOCK_DETECTED: avoiding_roadblock,
                CAR_DETECTED: tailing_car,
            },
            pedestrian_crossing: {
                PEDESTRIAN_WAITING: pedestrian_crossing,
                PEDESTRIAN_CROSSED: lane_following,
            },
            avoiding_roadblock: {
                TRY_AVOID_ROADBLOCK: avoiding_roadblock,
                PEDESTRIAN_CROSSED: lane_following,
                IN_CONTINUE_LANE: continue_lane,
            },
            continue_lane: {
                WAITING_OBSTACLE: continue_lane,
                OBSTACLE_GONE: lane_following,
            },
            tailing_car: {
                IN_CONTINUE_LANE: continue_lane,
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
            END_EVENT: self.on_end,
            CAR_OVERTAKEN: self.on_car_overtaken,
            ROADBLOCK_AVOIDED: self.on_roadblock_avoided,
            IF_MOVING: self.on_car_moving,
            IF_STATIC: self.on_car_static,

            CONTINUE_LANE_FOLLOWING: self.on_lane_following,
            ULTRA_SOUND_EMERGENCY: self.on_ultra_sound_emergecy,

            ###############################
            STOP_LINE_APPROACH_DISTANCE_THRESHOLD: self.on_stop_line_approach_distance_threshold,

            INTERSECTION_STOP: self.on_intersection_stop,
            STOPLINE_TIMEOUT: self.on_stopline_timeout,
            STOPLINE_WAITING: self.on_stopline_waiting,
            
            ###############################
            SIGN_DISTANCE_THRESHOLD: self.on_sign_distance_threshold,

            STOP_SIGN_DETECTED: self.on_stop_sign_detected,
            WAITING_STOP: self.on_waiting_stop,
            TIMEOUT_STOP: self.on_timeout_stop,

            PARKING_SIGN_DETECTED: self.on_parking,
            TRY_PARKING: self.on_try_parking,
            PARKING_COMPLETED: self.on_parking_completed,
            
            HIGHWAY_ENTRY_SIGN_DETECTED: self.on_highway_entry_sign_detected,
            IN_HIGHWAY: self.on_highway,
            HIGHWAY_END_SIGN_DETECTED: self.on_highway_end_sign_detected,
        
            CROSSWALK_SIGN_DETECTED: self.on_crosswalk_sign_detected,
            CROSSWALK_WAITING: self.on_crosswalk_waiting,
            CROSSWALK_TIMEOUT: self.on_crosswalk_timeout,

            ONE_WAY_SIGN_DETECTED: self.on_one_way_sign_detected,
            ROUNDABOUT_SIGN_DETECTED: self.on_roundabout_way_sign_detected,
            NO_ENTRY_SIGN_DETECTED: self.on_no_entry_sign_detected,
           
            FINAL_NODE_REACHED: self.on_final_node_reached,

            INTERSECTION_PRIORITY: self.on_intersection_priority,
            PRIORITY_SIGN_DETECTED: self.on_priority_sign_detected,

            ###############################
            TRAFFIC_LIGHT_DETECTED:self.on_traffic_light_detected,
            YELLOW_LIGHT_DETECTED: self.on_yellow_light_detected,
            YELLOW_LIGHT_WAITING: self.on_yellow_light_waiting,
            YELLOW_LIGHT_FINISHED: self.on_yellow_light_finished,
            GREEN_LIGHT_DETECTED: self.on_green_light_detected,
            RED_LIGHT_DETECTED: self.on_red_light_detected,
            RED_LIGHT_FINISHED: self.on_red_light_finished,
            RED_LIGHT_WAITING: self.on_red_light_waiting,


            ###############################
            OBSTACLE_DISTANCE_THRESHOLD: self.on_obstacle_distance_threshold,

            PEDESTRIAN_DETECTED: self.on_pedestrian_detected,
            PEDESTRIAN_WAITING: self.on_pedestrian_waiting,
            PEDESTRIAN_CROSSED: self.on_pedestrian_crossed,


            ROADBLOCK_DETECTED: self.on_roadblock_detected,
            ROADBLOCK_AVOIDING: self.on_roadblock_avoiding,
            ROADBLOCK_AVOIDED: self.on_roadblock_avoided,

            CAR_DETECTED: self.on_car_detected,
            CAR_MOVING: self.on_car_moving,
            CAR_STATIC: self.on_car_static,
            CAR_OVERTAKEN: self.on_car_overtaken,

            IN_CONTINUE_LANE: self.on_continue_lane,
            CONTINUE_LANE_DRIVING: self.on_continue_lane_driving,
            OBSTACLE_WAITING: self.on_obstacle_waiting,
            OBSTACLE_GONE: self.on_obstacle_gone
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
        self.init_stopline_time = 0
        # CROSSWALK
        self.timeout_crosswalk = 6
        self.in_crosswalk = False

        # PRIORITY
        self.priority = False

        # TRAFFIC LIGHT
        self.traffic_state = RED

        # SIGNS
        self.signs_detected = None
        self.current_sign = None

        # OBSTACLES
        self.obstacles_detected = None
        self.current_obstacle = None
        
        # PEDESTRIAN
        self.pedestrian_in_street = None

        # PATH LOCAL TRACKING
        self.final_node_reached = False

        # LANE FOLLOWING
        self.e2 = 0
        self.e3 = 0
        self.stop_line = False
        self.stopline_valid_distance = 0

        # STANDART VARIABLES
        self.current_state = lane_following #start_state
        self.speed_blocked = False
        self.current_speed = None
        self.current_steer = None
        self.current_ultra_values = None

        
    #===================== STATE HANDLING =====================#
    def change_state(self, event):
        """Cambia el estado basándose en el evento."""
        if self.current_state in self.state_transitions and event in self.state_transitions[self.current_state]:
            new_state = self.state_transitions[self.current_state][event]
            print(f"Cambiando de estado: {self.current_state} -> {new_state} por evento: {event}")
            self.event_methods[event]()
            self.current_state = new_state
        else:
            print(f"No se puede cambiar al estado con el evento: {event} desde {self.current_state}")
    #===================== EVENT HANDLING =====================#
    def handle_events(self, e2, e3, curv, desired_speed, signs_detected, obstacles_detected, stopline_valid_distance, ultra_values):
        self.e2 = e2 if e2 is not None else self.e2
        self.e3 = e3 if e3 is not None else self.e3
        self.curv = curv if curv is not None else self.curv
        self.desired_speed = desired_speed if desired_speed is not None else self.desired_speed
        self.signs_detected = signs_detected if signs_detected is not None else self.signs_detected
        self.obstacles_detected = obstacles_detected if obstacles_detected is not None else self.obstacles_detected
        self.stopline_valid_distance = stopline_valid_distance if stopline_valid_distance is not None else self.stopline_valid_distance
        self.current_ultra_values = ultra_values if ultra_values is not None else self.current_ultra_values
        # TODO: separar los objetos detectados en obstaculos y señales
        if self.current_ultra_values > 10000:
            # seteo el state lane_following para no agregar en cada state la relacion state-> evento/ULTRA_SOUND_EMERGENCY -> classifying_obstacle 
            self.current_state = lane_following
            self.change_state(ULTRA_SOUND_EMERGENCY)
        elif self.current_state == lane_following:
            if signs_detected and any(valid_distance for _, valid_distance in signs_detected):
                self.change_state(SIGN_DISTANCE_THRESHOLD)
            elif obstacles_detected and any(valid_distance for _, valid_distance in obstacles_detected):
                self.change_state(OBSTACLE_DISTANCE_THRESHOLD)
            elif self.stopline_valid_distance:
                self.change_state(STOP_LINE_APPROACH_DISTANCE_THRESHOLD)
            else:
                self.change_state(CONTINUE_LANE_FOLLOWING)
        elif self.current_state == traffic_light_state:
            if self.traffic_state == RED:
                self.change_state(RED_LIGHT_DETECTED)
            elif self.traffic_state == YELLOW:
                self.change_state(YELLOW_LIGHT_DETECTED)
            elif self.traffic_state == GREEN:
                self.change_state(GREEN_LIGHT_DETECTED)
        elif self.current_state == red_state:
            if self.traffic_red_light == True:
                self.change_state(RED_LIGHT_WAITING)
            else:
                self.change_state(RED_LIGHT_FINISHED)
        elif self.current_state == approaching_stop_line:
            if self.priority == True:
                self.priority = False
                self.change_state(INTERSECTION_PRIORITY_EVENT)
            else:
                self.change_state(INTERSECTION_STOP)
        elif self.current_state == stopline_state:
            if self.stopline_valid_distance:
                self.change_state(STOPLINE_WAITING)
            else:
                self.change_state(STOPLINE_TIMEOUT)
        elif self.current_state == classifying_signal:
            if self.current_sign == STOP_SIGN:
                self.change_state(STOP_SIGN_DETECTED)
            elif self.current_sign == PARKING_SIGN:
                self.change_state(PARKING_SIGN_DETECTED)
            elif self.current_sign == CROSSWALK_SIGN:
                self.change_state(CROSSWALK_SIGN_DETECTED)
            elif self.current_sign == PRIORITY_SIGN:
                #TODO: talvez desactivar la deteccion de autos con una variable sign_name == "CAR" and valid_distance and priority == False
                # pero podria generar problemas al tener un auto de frente y con velocidad menor a la nuestra
                self.change_state(PRIORITY_SIGN_DETECTED)
            elif self.current_sign == HIGHWAY_ENTRY_SIGN:
                self.change_state(PARKING_SIGN_DETECTED)
            elif self.current_sign == ONE_WAY_SIGN:
                #TODO: desactivar el intento de traspasar autos 
                self.change_state(ONE_WAY_SIGN_DETECTED)
            elif self.current_sign == ROUNDABOUT_SIGN:
                #TODO: activar seguimiento de nodos para movernos
                self.change_state(ROUNDABOUT_SIGN_DETECTED)
            elif self.current_sign == NO_ENTRY_SIGN:
                #TODO: tachar un nodo de la lista de nodos como no utilizable
                self.change_state(NO_ENTRY_SIGN_DETECTED)
            elif self.current_sign == PEDESTRIAN:
                #TODO: parar hasta que deje de detectar
                self.change_state(PEDESTRIAN_DETECTED)
            elif self.current_sign == CAR:
                #TODO: parar hasta que deje de detectar
                self.change_state(CAR_DETECTED)
            elif self.current_sign == ROADBLOCK:
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
        elif self.current_state == crosswalk_state:
            if self.in_crosswalk == True:
                self.change_state(WAITING_CROSSWALK)
            else:
                self.change_state(CROSSWALK_TIMEOUT)
        elif self.current_state == classifying_obstacle:
            if self.current_obstacle == PEDESTRIAN:
                self.change_state(PEDESTRIAN_DETECTED)
            elif self.current_obstacle == CAR:
                self.change_state(CAR_DETECTED)
            elif self.current_obstacle == ROADBLOCK:
                self.change_state(ROADBLOCK_DETECTED)
        elif self.current_state == pedestrian_crossing:
            if self.pedestrian_in_street == True:
                self.change_state(PEDESTRIAN_WAITING)
            else:
                self.change_state(PEDESTRIAN_CROSSED)
        elif self.current_state == tracking_local_path:
            self.change_state(FINAL_NODE_REACHED)
        
        return self.current_speed, self.current_steer
    
    def on_ultra_sound_emergecy(self):
        self.set_speed(0)
        self.set_steer(0)
        self.speed_blocked = False
        self.current_speed = None
        self.current_steer = None
        self.signs_detected = None
        self.obstacles_detected = None
        self.current_ultra_values = None
        self.current_obstacle = None
        self.current_sign = None
        self.pedestrian_in_street = None
        self.final_node_reached = False
        self.e2 = 0
        self.e3 = 0
        self.stop_line = False
        self.stopline_valid_distance = 0

    def on_roadmap_loaded(self): 
        path_file = "src/example/src/nodes_following/Competition_track_graph.graphml"

        self.track_graph = nx.read_graphml(path_file)
        # esto va para tracking_local_path, self.path serian los nodos a tomar
        #self.cx = [float(self.track_graph.nodes[node]["x"]) for node in self.path]
        #self.cy = [float(self.track_graph.nodes[node]["y"]) for node in self.path]
        # roundabout_event setea path y TRACKING_NODE ejecuta pure pursuit
    def on_end(self): 
        self.set_speed(0)
        self.set_steer(0)

    def on_lane_following(self): 
        speed, angle_ref = self.control_system.get_control(self.e2, self.e3, 0, self.desired_speed)
        angle_ref = np.rad2deg(angle_ref)
        self.set_steer((angle_ref - 6 ) * 10)
        self.set_speed(int(speed * 1000))

    
    def on_sign_distance_threshold(self):
        self.current_sign = next((sign for sign, valid_distance in self.signs_detected if valid_distance), None)
    def on_stop_sign_detected(self):
        # TODO: revisar valor de stop que recibe el modulo
        self.timeout_stop = time.time()
        self.set_speed(self.distance_module.handle_stop_signal_logic(self.signs_detected["STOP"], self.current_speed))
        self.speed_blocked = True
    def on_waiting_stop(self):
        current_time = time.time()  
        elapsed_time = current_time - self.timeout_stop 
        if elapsed_time >= self.stop_time:
            self.timeout_stop = -1
    def on_timeout_stop(self):
        self.speed_blocked = False 
        
    def on_parking(self):
        self.parking_start_time = time.time()  
        self.parking_step = 0 
        self.a=False
        self.b = False
        self.c = False
        self.d = False
    def on_try_parking(self): 
        current_time = time.time()  
        elapsed_time = current_time - self.parking_start_time 

        if self.parking_step == 0:  
            self.set_speed(200)
            self.set_steer(0)
            if elapsed_time >= self.parking_duration[0]:
                self.parking_step += 1
                self.parking_start_time = current_time  
        elif self.parking_step == 1:  
            self.set_speed(0)
            self.set_steer(240)
            if elapsed_time >= self.parking_duration[1]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 2:  
            self.set_speed(-100)
            self.set_steer(240)
            if elapsed_time >= self.parking_duration[2]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 3: 
            self.set_speed(-100)
            self.set_steer(0)
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 4: 
            self.set_speed(-100)
            self.set_steer(0)
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 5: 
            self.set_speed(0)
            self.set_steer(240)
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 6: 
            self.set_speed(0)
            self.set_steer(-240) 
            if elapsed_time >= self.parking_duration[3]:
                self.parking_step += 1
                self.parking_start_time = current_time
        elif self.parking_step == 7:
            self.set_speed(0)
            self.set_steer(0)
            if elapsed_time >= self.parking_duration[4]:
                self.parking_step = 0  
                self.parking_start_time = None 
                self.parking_end_time = current_time  
                self.on_parking_completed()  
    def on_parking_completed(self): 
        print("Secuencia de estacionamiento completada.")
        self.parking_end_time = -1
        #end time setear en None y en handle events consultar esta variable


    def on_highway_entry_sign_detected(self):
        self.highway_end = False
        self.set_speed(300)
    def on_highway(self):
        # TODO:revisa posicion actual, nodo actual y ve si cambia o no. al llegar al nodo final, levanta una bandera aunque ya deberia
        # haber detectado la senal de fin de highway POSIBLENTE NO SEA NECESARIO CONSULTAR NODOS
        # si da igual el tema de nodos, entonce esta funcion no sirve, cuando detecta, en on highway entry sign detected, aumenta velocidad, guardando la actual, y cuando detecta el fin, retoma la velocidad inicla
        if any(sign == HIGHWAY_END_SIGN and valid_distance for sign, valid_distance in self.signs_detected):
                self.on_highway_end_sign_detected()
        pass
    def on_highway_end_sign_detected(self):
        self.highway_end = True

    def on_crosswalk_sign_detected(self):
        self.timeout_crosswalk = time.time()
        self.set_speed(50)
        self.in_crosswalk = True
        self.speed_blocked = True
    def on_crosswalk_waiting(self):
        current_time = time.time()  
        elapsed_time = current_time - self.timeout_crosswalk 
        if elapsed_time >= self.stop_time:
            self.in_crosswalk = False
    def on_crosswalk_timeout(self):
        self.speed_blocked = False

    def on_traffic_light_detected(self):
        if any(sign == RED and valid_distance for sign, valid_distance in self.signs_detected):
                self.traffic_state = RED
        elif any(sign == YELLOW and valid_distance for sign, valid_distance in self.signs_detected):
                self.traffic_state = YELLOW
        elif any(sign == GREEN and valid_distance for sign, valid_distance in self.signs_detected):
                self.traffic_state = GREEN 
    def on_red_light_detected(self):
        self.traffic_red_light = True
        self.set_speed(0)
        self.speed_blocked = True
    def on_red_light_waiting(self):
        if any(sign == RED and valid_distance for sign, valid_distance in self.signs_detected):
            self.traffic_red_light = False
    def on_red_light_finished(self):
        self.speed_blocked = False
    def on_yellow_light_detected(self):
        self.set_speed(50)
        self.speed_blocked = True
    def on_yellow_light_waiting(self):
        if any(sign == RED and valid_distance for sign, valid_distance in self.signs_detected):
            self.on_yellow_light_finished()
    def on_yellow_light_finished(self):
        self.speed_blocked = False
    def on_green_light_detected(self):
        #TODO: cuando es verde no hay que hacer nada CREO
        pass
    
    def on_one_way_sign_detected(self):
        self.final_node_reached = False

        #TODO: inicializar nodo inicial, nodo a seguir y nodo final del path local tracking
        pass
    def on_roundabout_way_sign_detected(self):
        self.final_node_reached = False

        #TODO: inicializar nodo inicial, nodo a seguir y nodo final del path local tracking
        pass
    def on_no_entry_sign_detected(self):
        #TODO: inicializar nodo inicial, nodo a seguir y nodo final del path local tracking
        self.final_node_reached = False
        pass
    def on_tracking_node(self):
        # if actual_node == final_node: self.on_final_node_reached()
        # TODO: aca deberia setear el siguiente nodo a seguir modificando direccion
        pass
    def on_final_node_reached(self):
        self.final_node_reached = True

    def on_priority_sign_detected(self): 
        self.priority = True
    def on_stop_line_approach_distance_threshold(self):
        self.stop_line = True
        self.desired_speed = 0.1
     #TODO: asumo que espera por tiempo y despues revisa que no hay vehiculos para avanzar
    def on_intersection_priority(self):
        pass
    def on_intersection_stop(self):
        self.init_stopline_time = time.time()
        print ("car", self.init_stopline_time)
        self.set_speed(0)
        self.speed_blocked = True
    def on_stopline_waiting(self):
        current_time = time.time()  
        elapsed_time = current_time - self.init_stopline_time
        print("tiempos: ", current_time, elapsed_time, self.init_stopline_time) 
        if elapsed_time >= self.stop_time: #and any(sign == CAR and valid_distance for sign, valid_distance in self.signs_detected):
                self.stop_line = False
    def on_stopline_timeout(self):
        self.speed_blocked = False

    def on_obstacle_distance_threshold(self):
        self.current_obstacle = next((obstacle for obstacle, valid_distance in self.obstacles_detected if valid_distance), None)

    def on_pedestrian_detected(self):
        self.pedestrian_in_street = True
        self.set_speed(0)
        self.speed_blocked = True
    def on_pedestrian_waiting(self):
        if any(obstacle == PEDESTRIAN and valid_distance for obstacle, valid_distance in self.obstacles_detected):
            self.pedestrian_in_street = False
    def on_pedestrian_crossed(self):
        self.speed_blocked = False

    def on_roadblock_detected(self):
        pass
    def on_roadblock_avoiding(self):
        pass
    def on_roadblock_avoided(self):
        pass

    def on_car_detected(self):
        pass
    def on_car_moving(self):
        pass
    def on_car_static(self):
        pass
    def on_car_overtaken(self):
        pass

    def on_continue_lane(self):
        pass
    def on_continue_lane_driving(self):
        pass
    def on_obstacle_waiting(self):
        pass
    def on_obstacle_gone(self):
        pass
    #TODO: posiblemente en los siguientes casos hay que bloquear el cambio de velocidad o definir un rango:
    # amarillo en semaforo (desbloquea el speed en semaforo verde o rojo)
    # crosswalk (dsp de un rato desbloquea el speed)
    # highway (cuando detecta el final de highway )
    # hay que bloquear el cambio de velocidades

    # TODO: ademas en el amarillo, crosswalk, highway hay que llamar al modulo de control pero sin tener en cuenta la velocidad devuelta
    
    def set_speed(self, speed):
        if self.speed_blocked == False:
            self.current_speed = speed/100

    def set_steer(self, steer):
        if self.speed_blocked == False:
            self.current_steer = steer