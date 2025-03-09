import base64
import cv2
import numpy as np


def encode_image(frame, format = ".jpg"):
        _, serialEncodedImg = cv2.imencode(format, frame)
        serialEncodedImageData = base64.b64encode(serialEncodedImg).decode("utf-8")
        return serialEncodedImageData

def decode_image(frame):
        decoded_image_data = base64.b64decode(frame)
        nparr = np.frombuffer(decoded_image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame