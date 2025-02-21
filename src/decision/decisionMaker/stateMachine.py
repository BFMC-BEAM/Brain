from src.decision.distance.distance_module import DistanceModule
from src.decision.line_following.purepursuit import ControlSystem
from src.utils.constants import BRAINLESS, CONTROL_FOR_SIGNS
from src.utils.messages import message_handler_sender

ALWAYS_ON_ROUTINES = [CONTROL_FOR_SIGNS]
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
        self.control_system = ControlSystem()
        # INITIALIZE ROUTINES
        self.routines = {        
            CONTROL_FOR_SIGNS: Routine(CONTROL_FOR_SIGNS,  self.control_for_signs),         
        }
        # mat state
        self.state_transitions = {
            'start_state': {'ROADMAP_LOADED': 'lane_following'},
            'lane_following': {
                'OBSTACLE_DISTANCE_THRESHOLD': 'classifying_obstacle',
                'SIGN_DISTANCE_THRESHOLD': 'classifying_sign',
                'END_EVENT': 'end_state',
                'STOP_LINE_APPROACH_DISTANCE_THRESHOLD': 'approaching_stop_line',
                'PARKING_EVENT': 'parking_state',
                'CONTINUE_LANE_FOLLOWING': 'lane_following',
            },
            'classifying_sign': {'STOP_SIGN_DETECTED': 'stop_car'},
            'stop_car': {'TIMEOUT_STOP': 'lane_following'},
            'parking_state': {'COMPLETED': 'lane_following'},
            'waiting_for_pedestrian': {'PEDESTRIAN_TIMEOUT': 'lane_following'},
            'overtaking_moving_car': {'CAR_OVERTAKEN': 'lane_following'},
            'overtaking_static_car': {'CAR_OVERTAKEN': 'lane_following'},
            'avoiding_roadblock': {'ROADBLOCK_AVOIDED': 'lane_following'},
            'classifying_obstacle': {
                'OBSTACLE=PEDESTRIAN': 'waiting_for_pedestrian',
                'OBSTACLE=CAR': 'tailing_car',
                'OBSTACLE=ROADBLOCK': 'avoiding_roadblock',
            },
            'tailing_car': {
                'IF_OBSTACLE_TOO_FAR': 'lane_following',
                'IF_MOVING': 'overtaking_moving_car',
                'IF_STATIC': 'overtaking_static_car',
            },
            'approaching_stop_line': {
                'CROSSWALK_EVENT': 'crosswalk_navigation',
                'INTERSECTION_TRAFFIC_LIGHT_EVENT': 'waiting_for_green',
                'INTERSECTION_STOP_EVENT': 'waiting_at_stopline',
                'JUNCTION_EVENT': 'intersection_navigation',
                'INTERSECTION_PRIORITY_EVENT': 'intersection_navigation',
                'ROUNDABOUT_EVENT': 'roundabout_navigation',
            },
            'crosswalk_navigation': {'TIMEOUT_CROSSWALK': 'lane_following'},
            'roundabout_navigation': {'ALWAYS': 'tracking_local_path'},
            'tracking_local_path': {'END_OF_LOCAL_PATH': 'lane_following'},
            'intersection_navigation': {'ALWAYS': 'tracking_local_path'},
            'waiting_at_stopline': {'TIMEOUT_STOPLINE': 'intersection_navigation'},
            'waiting_for_green': {'SEMAPHORE_GREEN': 'intersection_navigation'}
        }
        
        self.event_methods = {
            'ROADMAP_LOADED': self.on_roadmap_loaded,
            'OBSTACLE_DISTANCE_THRESHOLD': self.on_obstacle_detected,
            'END_EVENT': self.on_end,
            'STOP_LINE_APPROACH_DISTANCE_THRESHOLD': self.on_stop_line_approach,
            'PARKING_EVENT': self.on_parking,
            'NOTHING_LANE_FOLLOWING': self.on_lane_following,
            'STOP_SIGN_DETECTED': self.on_stop_sign,
            'TIMEOUT_STOP': self.on_timeout_stop,
            'COMPLETED': self.on_parking_completed,
            'PEDESTRIAN_TIMEOUT': self.on_pedestrian_timeout,
            'CAR_OVERTAKEN': self.on_car_overtaken,
            'ROADBLOCK_AVOIDED': self.on_roadblock_avoided,
            'OBSTACLE=PEDESTRIAN': self.on_pedestrian_detected,
            'OBSTACLE=CAR': self.on_car_detected,
            'OBSTACLE=ROADBLOCK': self.on_roadblock_detected,
            'IF_OBSTACLE_TOO_FAR': self.on_obstacle_too_far,
            'IF_MOVING': self.on_car_moving,
            'IF_STATIC': self.on_car_static,
            'CROSSWALK_EVENT': self.on_crosswalk_event,
            'TIMEOUT_CROSSWALK': self.on_timeout_crosswalk,
            'INTERSECTION_TRAFFIC_LIGHT_EVENT': self.on_waiting_for_green,
            'INTERSECTION_STOP_EVENT': self.on_waiting_at_stopline,
            'JUNCTION_EVENT': self.on_intersection_navigation,
            'INTERSECTION_PRIORITY_EVENT': self.on_intersection_navigation,
            'ROUNDABOUT_EVENT': self.on_roundabout_navigation,
            'ALWAYS': self.on_always,
            'END_OF_LOCAL_PATH': self.on_end_of_local_path,
            'TIMEOUT_STOPLINE': self.on_timeout_stopline,
            'SEMAPHORE_GREEN': self.on_semaphore_green,
        }
        
        
        self.current_state = 'lane_following' #start_state
        self.current_speed = "0"
        self.current_steer = "0"

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
        


    def handle_events(self, act_deviation, num_lines_detected, objects_detected, current_speed, current_steer, intersection):
        self.act_deviation = act_deviation
        self.num_lines_detected = num_lines_detected
        self.objects_detected = objects_detected
        self.current_speed = current_speed
        self.current_steer = current_steer
        if self.current_state == 'lane_following':
            
            if intersection == 1:
                self.change_state('STOP_SIGN_DETECTED')
            else:
                self.change_state("CONTINUE_LANE_FOLLOWING")
        elif self.current_state == 'classifying_sign':
            for sign_name, valid_distance in objects_detected:
                if sign_name == "STOP" and valid_distance:
                    self.change_state("STOP_SIGN_DETECTED")

        elif self.current_state == 'classifying_obstacle':
            for object_name, valid_distance in objects_detected:
                if object_name == "PEDESTRIAN" and valid_distance:
                    self.change_state("OBSTACLE=PEDESTRIAN")
                elif object_name == "CAR" and valid_distance:
                    self.change_state("OBSTACLE=CAR")
                elif object_name == "ROADBLOCK" and valid_distance:
                    self.change_state("OBSTACLE=ROADBLOCK")
        return self.current_speed, self.current_steer
     
    def on_roadmap_loaded(self): pass
    def on_obstacle_detected(self): pass
    def on_end(self): pass
    def on_stop_line_approach(self): pass
    def on_parking(self): pass
    def on_lane_following(self): 
        self.current_steer = str(self.controlSystem.adjust_direction(self.act_deviation, self.direction) * 10)
    def on_stop_sign_detected(self):
        # TODO: revisar valor de stop que recibe el modulo
        self.current_speed = self.distanceModule.handle_stop_signal_logic(self.objects_detected["STOP"], self.current_speed)
    def on_timeout_stop(self):
        pass
    def on_parking_completed(self): pass
    def on_pedestrian_timeout(self): pass
    def on_car_overtaken(self): pass
    def on_roadblock_avoided(self): pass
    def on_pedestrian_detected(self): pass
    def on_car_detected(self): pass
    def on_roadblock_detected(self): pass
    def on_obstacle_too_far(self): pass
    def on_car_moving(self): pass
    def on_car_static(self): pass
    def on_crosswalk_event(self): pass
    def on_timeout_crosswalk(self): pass
    def on_waiting_for_green(self): pass
    def on_waiting_at_stopline(self): pass
    def on_intersection_navigation(self): pass
    def on_roundabout_navigation(self): pass
    def on_always(self): pass
    def on_end_of_local_path(self): pass
    def on_timeout_stopline(self): pass
    def on_semaphore_green(self): pass

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
