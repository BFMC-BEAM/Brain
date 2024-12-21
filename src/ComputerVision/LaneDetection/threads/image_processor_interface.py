from abc import ABC, abstractmethod

class ImageProcessorInterface(ABC):
    """
    Interfaz para procesadores de imágenes.
    """
    @abstractmethod
    def process_image(self, cv_image):
        """
        Procesa la imagen y realiza alguna acción (detección de curvas, líneas o de algun modelo que vayamos a testear).
        :param cv_image: Imagen en formato OpenCV.
        """
        pass