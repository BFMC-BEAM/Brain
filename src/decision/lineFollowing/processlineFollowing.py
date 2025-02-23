if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.decision.lineFollowing.threads.threadlineFollowing import threadlineFollowing

class processlineFollowing(WorkerProcess):
    """This process handles lineFollowing.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processlineFollowing, self).__init__(self.queuesList)

    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processlineFollowing, self).run()

    def _init_threads(self):
        """Create the lineFollowing Publisher thread and add to the list of threads."""
        lineFollowingTh = threadlineFollowing(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(lineFollowingTh)
