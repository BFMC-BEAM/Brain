import time

class DistanceModule():
    def __init__(self):
        self.min_distance = 5
        self.speed_increment = 50
        self.mult_increment = 0.4
        self.ignore_stop_signal_until = 0
        self.delay_stop_signal = 0
        self.start_stop_signal_logic = False
        #self.previous_speed = "0"
        
    def check_distance(self, ultraVals, currentSpeed, currentSteer):

        mult_distance = self.min_distance * self.get_multiplier(currentSpeed)
        # if ultraVals is not None:
        #     if ultraVals["top"] < mult_distance and int(currentSpeed) > 0:
        #         return ("0",currentSteer) #stop the vehicle if front distance is less than 30 cm
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
    
    def handle_stop_signal_logic(self, objectDetection, actualSpeed):
        current_time = time.time()
        decidedSpeed = actualSpeed

        if objectDetection == "stop_signal" and current_time > self.ignore_stop_signal_until and not self.start_stop_signal_logic:
            self.start_stop_signal_logic = True
            #self.previous_speed = decidedSpeed                          # save actualSpeed before stop
            self.delay_stop_signal = current_time + 3                   # stop for 3 seconds
            self.ignore_stop_signal_until = self.delay_stop_signal + 10 # ignore stop signal for 10 seconds

        if self.start_stop_signal_logic:
            if current_time > self.delay_stop_signal:
                self.start_stop_signal_logic = False
                #print(f"stop signal logic.")
            else:
                decidedSpeed = "0"

        return decidedSpeed