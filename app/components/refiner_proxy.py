import multiprocessing as mp
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any

import yaml
from torch import Tensor

from app import REFINER_PROPERTY_TYPES
from app.components.parallelism import start_process, start_thread, stop_event
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene


def sanitise(data: Any) -> Any:
    """
    Convert data to a serialisable format.
    """
    if isinstance(data, Tensor):
        return data.detach().cpu()
    elif isinstance(data, list):
        return [sanitise(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitise(item) for item in data)
    elif isinstance(data, dict):
        return {key: sanitise(value) for key, value in data.items()}
    elif isinstance(data, Crystal):
        return data.to_dict(include_buffers=True)
    elif isinstance(data, Scene):
        return data.to_dict()
    else:
        return data


def unsanitise(data: Any) -> Any:
    """
    Convert data back from a serialisable format to the original format.
    """
    if isinstance(data, list):
        return [unsanitise(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(unsanitise(item) for item in data)
    elif isinstance(data, dict):
        try:
            return Crystal.from_dict(data)
        except Exception:
            try:
                return Scene.from_dict(data)
            except Exception:
                pass
        return {key: unsanitise(value) for key, value in data.items()}
    else:
        return data


def refiner_worker(
        refiner_args: RefinerArgs,
        output_dir: Path,
        queues: dict,
):
    """
    Process worker for the Refiner instance.
    """
    refiner = Refiner(args=refiner_args, output_dir=output_dir)
    requests = []
    callback_signals = {}

    def wrap_callables(channel: str, args: list) -> list:
        wrapped_args = []
        response_queue = queues[channel]['response']
        for arg in args:
            # Wrap the callback in a function that puts the result in the response queue
            if isinstance(arg, dict) and 'type' in arg and arg['type'] == 'callback':
                def callback(*callback_args):
                    assert channel == arg['channel']
                    response_queue.put({'type': 'callback_result', 'channel': channel, 'data': callback_args})

                    # Check for a callback signal to return
                    if channel in callback_signals:
                        sig = callback_signals[channel]
                        del callback_signals[channel]
                        return sig

                wrapped_args.append(callback)
            else:
                wrapped_args.append(arg)
        return wrapped_args

    def handle_attribute_request(channel: str, attr_name: str, args: list):
        response_queue = queues[channel]['response']
        if hasattr(refiner, attr_name):
            attr = getattr(refiner, attr_name)

            # Property access
            if channel == 'properties':
                result = attr

            # Method call
            elif callable(attr):
                result = attr(*unsanitise(args))

            else:
                response_queue.put({'type': 'error', 'message': f'Invalid attribute/method: {attr_name}'})
                return

            # Put the result in the response queue
            response_queue.put({'type': 'result', 'data': sanitise(result)})
        else:
            response_queue.put({'type': 'error', 'message': f'Invalid attribute/method: {key}'})

    def handle_check_types_request(args: list):
        response_queue = queues['check_types']['response']
        assert len(args) == 1, f'Expected 1 argument to check type of, got {len(args)}'
        check_key = args[0]
        if hasattr(refiner, check_key):
            response_queue.put({
                'type': 'check_result',
                'check_key': check_key,
                'check_type': 'method' if callable(getattr(refiner, check_key)) else 'attribute'
            })
        else:
            response_queue.put({
                'type': 'error',
                'message': f'Invalid attribute/method: {check_key}'
            })

    def monitor_queue(channel: str, stop_event: threading.Event):
        nonlocal callback_signals, requests
        request_queue = queues[channel]['request']
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=1)
            except Empty:
                continue
            if request is None:
                time.sleep(1)
                continue

            # Process the request
            key, args = request

            # Check types requests
            if channel == 'check_types':
                handle_check_types_request(args)

            # Callback signals
            elif channel == 'callback_signals':
                callback_signals[key] = args

            # Property access requests
            elif channel == 'properties':
                handle_attribute_request(channel, key, [])

            # Method requests - run in the main process
            else:
                requests.append({'queue': channel, 'request': request})

            time.sleep(1)

    # Start monitoring the request queues
    for channel in queues.keys():
        thread = threading.Thread(target=monitor_queue, args=(channel, stop_event))
        start_thread(thread)

    # Process main-thread requests
    while True:
        if len(requests) == 0:
            time.sleep(1)
            continue

        req = requests.pop(0)
        channel, req = req['queue'], req['request']
        key, args = req

        # Method requests
        wrapped_args = wrap_callables(channel, args)
        handle_attribute_request(channel, key, wrapped_args)


class RefinerProxy:
    def __init__(
            self,
            args: RefinerArgs,
            output_dir: Path,
    ):
        self.args = args
        self.output_dir = output_dir
        self.active_methods = {}
        self.return_values = {}
        self._callbacks = {}
        self._init_property_types_cache()

        # Set up the channels
        self.queues = {
            channel: {
                'request': mp.Queue(),
                'response': mp.Queue()
            }
            for channel in ['check_types', 'properties', 'callback_signals',
                            'set_anchors', 'set_initial_scene', 'make_initial_prediction',
                            'train']
        }

        # Start the worker process
        self.process = mp.Process(
            target=refiner_worker,
            args=(self.args, self.output_dir, self.queues),
        )
        start_process(self.process)

    def _init_property_types_cache(self):
        """
        Initialise the property types cache from the file.
        """
        self._property_types = {}
        if REFINER_PROPERTY_TYPES.exists():
            with open(REFINER_PROPERTY_TYPES, 'r') as f:
                self._property_types = yaml.safe_load(f) or {}

    def _get_attribute_type(self, name: str) -> str:
        """
        Get the type of the attribute - either "method" or "attribute".
        """
        # Check the cache
        if name in self._property_types:
            return self._property_types[name]

        # Send a check request to the worker
        request_queue = self.queues['check_types']['request']
        response_queue = self.queues['check_types']['response']
        request_queue.put(('check_type', [name]))
        response = response_queue.get()
        if response['type'] == 'error':
            raise AttributeError(response['message'])
        assert response['type'] == 'check_result' and response['check_key'] == name \
               and response['check_type'] in ['method', 'attribute']

        # Update the cache
        self._property_types[name] = response['check_type']
        with open(REFINER_PROPERTY_TYPES, 'w') as f:
            yaml.safe_dump(self._property_types, f)

        return self._property_types[name]

    def __getattr__(self, name: str):
        """
        Access a property or method from the Refiner instance running in a separate process.
        """
        attr_type = self._get_attribute_type(name)

        # Method call
        if attr_type == 'method':
            request_queue = self.queues[name]['request']
            response_queue = self.queues[name]['response']

            def wrapper(*args):
                processed_args = [self._wrap_callables_and_sanitise(name, arg) for arg in args]
                request_queue.put((name, processed_args))
                self.active_methods[name] = 1
                self.return_values[name] = None

                def monitor_queue():
                    while True:
                        response = response_queue.get()
                        if response['type'] == 'callback_result':
                            callback = self._callbacks[name]
                            callback(*response['data'])
                            continue

                        # Stop monitoring and mark method as inactive
                        self.active_methods[name] = 0
                        self.return_values[name] = None
                        if response['type'] == 'result':
                            self.return_values[name] = response['data']

                        # Stop monitoring
                        return

                thread = threading.Thread(target=monitor_queue)
                start_thread(thread)
                thread.join()
                return_val = self.return_values[name]
                self.return_values[name] = None
                return return_val

            return wrapper

        # Property access
        else:
            request_queue = self.queues['properties']['request']
            response_queue = self.queues['properties']['response']
            request_queue.put((name, []))
            response = response_queue.get()
            if response['type'] == 'error':
                raise AttributeError(response['message'])
            data = response['data']
            if name == 'crystal' and isinstance(data, dict):
                data = Crystal.from_dict(data)
            elif name == 'scene' and isinstance(data, dict):
                data = Scene.from_dict(data)
            return data

    def is_training(self) -> bool:
        """
        Check if the worker process is training.
        """
        if 'train' not in self.active_methods:
            return False
        return self.active_methods['train']

    def stop_training(self):
        """
        Send a stop signal to the worker process.
        """
        self.queues['callback_signals']['request'].put(('train', False))

    def _wrap_callables_and_sanitise(self, channel: str, arg: Any):
        """
        Wrap a callable in a dictionary that can be serialised or sanitise the argument.
        """
        if callable(arg):
            self._callbacks[channel] = arg
            return {'type': 'callback', 'channel': channel}
        else:
            return sanitise(arg)
