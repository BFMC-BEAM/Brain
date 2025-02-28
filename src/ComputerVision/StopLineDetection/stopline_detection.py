from time import time
import numpy as np
import cv2 as cv
from os.path import join, dirname, realpath

from src.ComputerVision.LaneDetection.image_processor_interface import ImageProcessorInterface

this_dir = dirname(realpath(__file__))
IMG_SIZE = (32,32) 
BEAM_STOP_LINE_DETECTION_PATH= join(this_dir,'stop_line_detector.onnx')
PREDICTION_OFFSET = 0.34

class StopLineDetectionProcessor(ImageProcessorInterface):
    def __init__(self):
        #lane following
        self.stop_line_detector = cv.dnn.readNetFromONNX(BEAM_STOP_LINE_DETECTION_PATH)
        self.lane_cnt = 0
        self.avg_stop_line_detection_time = 0
        
    def process_image(self, frame):
        start_time = time()
        blob = self.preprocess(frame)
        self.stop_line_detector.setInput(blob)
        output = self.stop_line_detector.forward()
        stopline_x = dist = output[0][0] + PREDICTION_OFFSET
        stopline_y = output[0][1]
        stopline_angle = output[0][2]
        self.est_dist_to_stop_line = dist
        stop_line_detection_time = 1000*(time()-start_time)
        self.avg_stop_line_detection_time = (self.avg_stop_line_detection_time*self.lane_cnt + stop_line_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        #print(f"stop_line_detection dist: {dist:.2f}, in {stop_line_detection_time:.2f} ms")
        return stopline_x, stopline_y, stopline_angle
    
    def preprocess(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]*(2/5)):,:]
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3,3), 0)
        frame = cv.resize(frame, IMG_SIZE)
        
        #frame_flip = cv.flip(frame, 1)
        #frames = np.stack((frame, frame_flip), axis=0)
        #blob = cv.dnn.blobFromImages(frames, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        return frame