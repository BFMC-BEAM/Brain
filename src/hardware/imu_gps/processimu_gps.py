if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.hardware.imu_gps.threads.threadimu_gps import threadimu_gps

class processimu_gps(WorkerProcess):
    """This process handles imu_gps.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processimu_gps, self).__init__(self.queuesList)

    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processimu_gps, self).run()

    def _init_threads(self):
        """Create the imu_gps Publisher thread and add to the list of threads."""
        imu_gpsTh = threadimu_gps(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(imu_gpsTh)
