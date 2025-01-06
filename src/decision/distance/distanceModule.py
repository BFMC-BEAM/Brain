class DistanceModule():
    def __init__(self):
        self.min_distance = 5
        
    def check_distance(self, ultraVals, currentSpeed, currentSteer):
        if ultraVals is not None:
            if ultraVals["top"] < self.min_distance and int(currentSpeed) > 0:
                return (0,currentSteer) #stop the vehicle if front distance is less than 30 cm
            elif ultraVals["bottom"] < self.min_distance and  int(currentSpeed) < 0:
                # return (0,currentSteer) commented until back ultra instalation
                pass
            return (currentSpeed,currentSteer)