class DistanceModule():
    def __init__(self):
        pass
    def check_distance(self, ultraVals, currentSpeed, currentSteer = None):
        if ultraVals is not None:
            if ultraVals["top"] < 20 and int(currentSpeed) > 0:
                return (0,None) #stop the vehicle if front distance is less than 30 cm
            elif ultraVals["bottom"] < 20 and  int(currentSpeed) < 0:
                # return (0,None) commented until back ultra instalation
                pass
            return (None,None)