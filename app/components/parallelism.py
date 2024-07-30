import multiprocessing as mp
import os
import signal
import threading
from multiprocessing import Process
from threading import Thread

from crystalsizer3d import logger

# Ensure that CUDA will work in subprocesses
mp.set_start_method('spawn', force=True)

# Global flag to signal threads to stop
stop_event = threading.Event()

# Keep track of processes and threads globally
_processes = []
_threads = []


def close_parallel_processes():
    """
    Terminate all processes and threads.
    """
    logger.info('Closing processes and threads...')

    # Signal threads to stop
    stop_event.set()

    # Terminate processes
    for process in _processes:
        if process.is_alive():
            logger.warning(f'Terminating process {process.pid}')
            os.kill(process.pid, signal.SIGTERM)
            process.join()  # Ensure process cleanup

    # Stop threads
    for thread in _threads:
        if thread.is_alive():
            logger.warning(f'Stopping thread {thread.name}')
            thread.join()  # Wait for thread to stop
            # thread.join(timeout=1)


def start_process(process: Process):
    """
    Start a process and add to the global list.
    """
    process.start()
    _processes.append(process)


def start_thread(thread: Thread):
    """
    Start a thread and add to the global list.
    """
    thread.start()
    _threads.append(thread)
