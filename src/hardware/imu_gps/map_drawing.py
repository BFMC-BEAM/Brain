import matplotlib.pyplot as plt
import networkx as nx
import cv2
import json
import time
import numpy as np
import matplotlib.transforms as transforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src.utils.constants import GPS_DATA_PATH, TRACK_GRAPH_PATH, TRACK_IMG_PATH, CAR_IMG_PATH


class MapDrawer():
    def __init__(self, graph):
        self.G = graph
        self.map, self.trajectory = self.create_map()
        self.x_data = []
        self.y_data = []
        self.yaw_data = []

        # Cargar la imagen del auto
        self.car_img = cv2.imread(CAR_IMG_PATH, cv2.IMREAD_UNCHANGED)
        self.car_img = cv2.cvtColor(self.car_img, cv2.COLOR_BGRA2RGBA)  # Convertir a RGBA para transparencia
        self.car_icon = OffsetImage(self.car_img, zoom=1)  # Ajusta el tamaño del auto

        self.car_ab = None  # Variable para el marcador del auto

    def create_map(self):
        # Cargar la imagen de la pista
        img = cv2.imread(TRACK_IMG_PATH, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:  # Si tiene canal alfa
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)  # Convertir a RGBA para matplotlib
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Si no tiene canal alfa, convertir a RGB


        # Obtener coordenadas de los nodos
        nodes = {n: (float(d["x"]), float(d["y"])) for n, d in self.G.nodes(data=True)}

        # Configurar la figura
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='none')
        ax.set_facecolor('none')

        ax.imshow(img, extent=[0, 6, 0, 6])  # Ajustar tamaño del mapa

        # Línea para la trayectoria GPS
        trajectory_line, = ax.plot([], [], 'c-', linewidth=2, alpha=0.7, label="GPS Trajectory")

        # Quitar labels y márgenes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)  # Oculta el borde
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Elimina márgenes

        return ax, trajectory_line

    def add_gps_data(self, x, y, yaw):
        self.x_data.append(x)
        self.y_data.append(y)
        self.yaw_data.append(yaw)
        self.update_pos(x, y, yaw)
        self.trajectory.set_data(self.x_data, self.y_data)

    def update_pos(self, x, y, yaw):
        if self.car_ab:
            self.car_ab.remove()  # Elimina el auto anterior

        # Rotar la imagen correctamente y volver a crear OffsetImage
        rotated_car = self.rotate_image(self.car_img, np.degrees(yaw))
        rotated_icon = OffsetImage(rotated_car, zoom=0.75)  # Convertir en objeto OffsetImage

        # Crear nueva anotación con la imagen rotada
        self.car_ab = AnnotationBbox(rotated_icon, (x, y), frameon=False)
        self.map.add_artist(self.car_ab)

    def rotate_image(self, image, angle):
        """ Rota la imagen del auto manteniendo la transparencia. """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Matriz de rotación
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return rotated

    def get_current_map(self):
        self.map.figure.canvas.draw()
        img_array = np.array(self.map.figure.canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)  # Convertir a formato OpenCV (BGR)
        img_bgr = cv2.resize(img_bgr,(400,400))
        return img_bgr
