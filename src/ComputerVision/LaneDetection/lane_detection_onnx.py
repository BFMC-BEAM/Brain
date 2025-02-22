from time import time
import numpy as np
import cv2 as cv
from os.path import join, dirname, realpath

from src.ComputerVision.LaneDetection.image_processor_interface import ImageProcessorInterface

this_dir = dirname(realpath(__file__))
IMG_SIZE = (32,32) 
BEAM_LANE_KEEPER_PATH= join(this_dir,'lane_keeper.onnx')
DISTANCE_POINT_AHEAD = 0.35
LK_CORRECTION = 1.1

class LaneDetectionProcessor(ImageProcessorInterface):
    def __init__(self):
        #lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(BEAM_LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0
        self.last_frame = None
        
    def process_image(self, frame):
        self.last_frame = frame
        self.preprocess(frame)
        start_time = time()
        blob = self.preprocess(frame)
        self.lane_keeper.setInput(blob)
        out = self.lane_keeper.forward() * LK_CORRECTION #### NOTE: MINUS SIGN IF OLD NET
        #print(out)
        e2, e3 = out[0]

        #calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])
        # print(f"est_point_ahead: {est_point_ahead}")
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_detection_time = (self.avg_lane_detection_time*self.lane_cnt + lane_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        return e2, e3, est_point_ahead
    
    def preprocess(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]*(1/3)):,:]
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3,3), 0)
        frame = cv.resize(frame, IMG_SIZE)
        
        #frame_flip = cv.flip(frame, 1)
        #frames = np.stack((frame, frame_flip), axis=0)
        #blob = cv.dnn.blobFromImages(frames, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        return frame