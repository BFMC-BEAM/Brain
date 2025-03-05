import matplotlib.pyplot as plt
import networkx as nx
import cv2
import json
import time
import numpy as np

from src.utils.constants import GPS_DATA_PATH, TRACK_GRAPH_PATH, TRACK_IMG_PATH



class MapDrawer():
    def __init__(self, graph):
        self.G = graph
        self.map, self.trajectory = self.create_map()
        self.x_data = []
        self.y_data = []
    
    def create_map(self):
        # Cargar la imagen de la pista
        img = cv2.imread(TRACK_IMG_PATH, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib

        # Obtener coordenadas de los nodos
        nodes = {n: (float(d["x"]), float(d["y"])) for n, d in self.G.nodes(data=True)}

        # Configurar la figura
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, extent=[0, 6, 0, 6])  # Mantener proporciones del mapa

        # Dibujar el grafo (nodos y aristas)
        for edge in self.G.edges():
            x_values = [nodes[edge[0]][0], nodes[edge[1]][0]]
            y_values = [nodes[edge[0]][1], nodes[edge[1]][1]]
            ax.plot(x_values, y_values, 'g-', linewidth=2)  # Líneas verdes para las aristas

        for node, (x, y) in nodes.items():
            ax.plot(x, y, 'ro', markersize=5)  # Puntos rojos para los nodos

        # Línea para la trayectoria GPS
        trajectory_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label="GPS Trajectory")

        # Quitar labels y márgenes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)  # Oculta el borde
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Elimina márgenes

        return ax, trajectory_line


    def add_gps_data(self, x, y):
        self.x_data.append(x)
        self.y_data.append(y)
        self.trajectory.set_data(self.x_data, self.y_data)
    
    def get_current_map(self):
        self.map.figure.canvas.draw()
        img_array = np.array(self.map.figure.canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)  # Convertir a formato OpenCV (BGR)
        return img_bgr

