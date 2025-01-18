import time

class DistanceModule():
    def __init__(self):
        self.min_distance = 5
        self.speed_increment = 50
        self.mult_increment = 0.4
        self.ignore_stop_signal_until = 0
        self.delay_stop_signal = 0
        self.start_stop_signal_logic = False
        self.start_previous_speed_logic = False
        self.previous_speed = 0
        
    def check_distance(self, ultraVals, currentSpeed, currentSteer):
        mult_distance = self.min_distance * self.get_multiplier(currentSpeed)
        if ultraVals is not None:
            if ultraVals["top"] < mult_distance and int(currentSpeed) > 0:
                return ("0",currentSteer) #stop the vehicle if front distance is less than 30 cm
            #elif ultraVals["bottom"] < self.min_distance and  int(currentSpeed) < 0:
                # return ("0",currentSteer) commented until back ultra instalation
                #pass
        return (currentSpeed,currentSteer)
    
    def check_stop_signal(self, objectDetection, currentSpeed):
        if objectDetection == "stop_signal":
            return("0")
        return (currentSpeed)


    def get_multiplier(self, currentSpeed):
        absSpeed= abs(int(currentSpeed))
        mult = 1
        if absSpeed > self.speed_increment:
            mult = (absSpeed / self.speed_increment) * self.mult_increment  + 1
        
        return mult
    
    def handle_stop_signal_logic(self, objectDetection, decidedSpeed):
        current_time = time.time()

        if objectDetection == "stop_signal" and current_time > self.ignore_stop_signal_until and not self.start_stop_signal_logic and not self.start_previous_speed_logic:
            self.start_stop_signal_logic = True
            self.previous_speed = decidedSpeed                          # Guardar la velocidad antes de detener el auto
            self.delay_stop_signal = current_time + 3                   # Tiempo de detención
            self.ignore_stop_signal_until = self.delay_stop_signal + 10 # Ignorar la señal de stop por 10 segundos
            print(f"Entered stop signal logic:")
            print(f"  self.delay_stop_signal: {self.delay_stop_signal}")
            print(f"  self.ignore_stop_signal_until: {self.ignore_stop_signal_until}")
            print(f"  self.previous_speed: {self.previous_speed}")

        if self.start_stop_signal_logic:
            if current_time > self.delay_stop_signal:
                self.start_stop_signal_logic = False
                self.start_previous_speed_logic = True
                print(f"stop signal logic.")
            else:
                decidedSpeed = "0"

        if self.start_previous_speed_logic:
            if current_time > self.ignore_stop_signal_until:
                self.start_previous_speed_logic = False
                print(f"stop previous speed logic.")
            else:
                decidedSpeed = self.previous_speed
            

        return decidedSpeed