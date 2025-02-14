
from ComputerVision.LaneDetection.lane_detection import LaneDetectionProcessor
from ComputerVision.ObjectDetection.object_detection import ObjectDetectionProcessor
from utils.constants import BRAINLESS, CONTROL_FOR_SIGNS
from utils.messages import messageHandlerSender

ALWAYS_ON_ROUTINES = [CONTROL_FOR_SIGNS]
LANE_FRAME_SKIP = 1
SIGNAL_FRAME_SKIP = 10


class State():
    def __init__(self, name=None, method=None):
        self.name = name
        self.method = method
    def __str__(self):
        return self.name.upper() if self.name is not None else 'None'
    def run(self):
        self.method()

class Routine():
    def __init__(self, name, method, activated=False):
        self.name = name
        self.method = method
        self.active = activated
    def __str__(self):
        return self.name
    def run(self):
        self.method()
class Event():
    pass

class StateMachine():
    def __init__(self, lane_processor: LaneDetectionProcessor, 
                 sign_processor: ObjectDetectionProcessor, 
                 direction :messageHandlerSender, 
                 deviation:messageHandlerSender,
                 ObjectDetection_Type:messageHandlerSender,
                 Lines: messageHandlerSender, #TODO: modificar nombre
                 FPS = 30
                 ):
          self.lane_processor = lane_processor
          self.sign_processor = sign_processor
          self.direction = direction
          self.deviation = deviation
          self.ObjectDetection_Type = ObjectDetection_Type
          self.lines = Lines
          self.frame = None
          self.act_deviation = 0
          self.curr_sign = "no_signal"
          self.act_lines = -1 # contador de lineas detectadas, 0 nada, 1 si detecto izq o der, 2 normal
          self.act_direction = "straight"
          self.curr_state = State(BRAINLESS, self.brainless)
          self.fps = FPS
          self.frame_count = 0
        # INITIALIZE ROUTINES
          self.routines = {        
            CONTROL_FOR_SIGNS: Routine(CONTROL_FOR_SIGNS,  self.control_for_signs),         
        }
    #===================== STATES =====================#
    def brainless(self):
        # SKIP FRAMES IF NEEDED
        if not self.frame_count % LANE_FRAME_SKIP != 0:
            return
        self.frame = self.lane_processor.process_image(self.frame)
        ret = self.lane_processor.get_parameters(self.act_deviation)
        new_cant_lines = self.lane_processor.get_lines()
        if ret[0] != -1000:
            self.direction.send(ret[1])  # Enviar dirección
            self.deviation.send(ret[0])  # Enviar desviación
            self.act_deviation = ret[0]
            print(f'direction: {ret[1]}')
        if new_cant_lines != self.act_lines:
            self.lines.send(new_cant_lines)
            self.act_lines = new_cant_lines

    #===================== ROUTINES =====================#
    def control_for_signs(self):
        # SKIP FRAMES IF NEEDED
        if not self.frame_count % SIGNAL_FRAME_SKIP != 0:
            return
        out, valid_distance = self.sign_processor.process_image(self.frame)
        self.frame = out
        if not valid_distance:
            self.ObjectDetection_Type.send("stop_signal")
            self.curr_sign = "stop_signal"
        else:
            self.ObjectDetection_Type.send("no_signal")
            self.curr_sign = "no_signal"

    #===================== STATE MACHINE MANAGEMENT =====================#
    def run(self, frame):
        """Método principal que se ejecuta en cada iteración del ciclo."""
        self.update_frame(frame)
        self.run_current_state()
        self.run_active_routines()
        self.log_status()

    def run_current_state(self):
        self.curr_state.run()

    def update_frame(self, frame):
        """Actualiza el frame y el contador de frames."""
        self.frame = frame
        self.frame_count = (self.frame_count + 1) % self.fps  # Evita overflow

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
