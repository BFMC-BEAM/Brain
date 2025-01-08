class DistanceModule():
    def __init__(self):
        self.min_distance = 5
        self.speed_increment = 5
        self.mult_increment = 0.4
        
    def check_distance(self, ultraVals, currentSpeed, currentSteer):
        mult_distance = self.min_distance * self.get_multiplier(currentSpeed)
        if ultraVals is not None:
            if ultraVals["top"] < mult_distance and int(currentSpeed) > 0:
                return ("0",currentSteer) #stop the vehicle if front distance is less than 30 cm
            #elif ultraVals["bottom"] < self.min_distance and  int(currentSpeed) < 0:
                # return ("0",currentSteer) commented until back ultra instalation
                #pass
        return (currentSpeed,currentSteer)
    
    def get_multiplier(self, currentSpeed):
        absSpeed= abs(int(currentSpeed))
        mult = 1
        if absSpeed > self.speed_increment:
            mult = (absSpeed / self.speed_increment) * self.mult_increment  + 1
        return mult