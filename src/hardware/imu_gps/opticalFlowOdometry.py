import cv2
import numpy as np

class OpticalFlowOdometry:
    def __init__(self, scale_factor=0.01):
        self.prev_frame = None
        self.scale_factor = scale_factor
        self.position = np.array([4.0, 0.75, 0.0])  # (x, y, theta)
        self.K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]]) # Ejemplo de matriz de camara, debe ser calibrada.

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return self.position

        # Detectar puntos clave y calcular descriptores
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.prev_frame, None)
        kp2, des2 = orb.detectAndCompute(gray, None)

        # Coincidencia de puntos clave
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extraer puntos coincidentes
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calcular la matriz esencial
        E, mask = cv2.findEssentialMat(points1, points2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            self.prev_frame = gray
            return self.position

        # Descomponer la matriz esencial para obtener rotación y traslación
        _, R, t, mask = cv2.recoverPose(E, points1, points2, self.K, mask=mask)

        # Actualizar la posición
        if R is not None and t is not None:
            self.position[:2] += (R @ t).flatten()[:2] * self.scale_factor
            self.position[2] += np.arctan2(R[1, 0], R[0, 0])

        self.prev_frame = gray
        return self.position