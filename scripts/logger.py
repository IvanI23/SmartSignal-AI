import threading

class WorkflowLogger:
    def __init__(self):
        self._messages = []
        self._lock = threading.Lock()

    def log(self, message):
        with self._lock:
            self._messages.append(message)

    def clear(self):
        with self._lock:
            self._messages = []

    def get_log(self):
        with self._lock:
            return '\n'.join(self._messages) 