if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.ComputerVision.ObjectDetection.threads.threadObjectDetection import threadObjectDetection

class processObjectDetection(WorkerProcess):
    """This process handles LaneDetection.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False, yolo_path=None):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.yolo_path = yolo_path
        super(processObjectDetection, self).__init__(self.queuesList)

    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processObjectDetection, self).run()

    def _init_threads(self):
        """Create the ObjectDetection Publisher thread and add to the list of threads."""
        ObjectDetectionTh = threadObjectDetection(
            self.queuesList, self.logging, self.debugging, self.yolo_path
        )
        self.threads.append(ObjectDetectionTh)
