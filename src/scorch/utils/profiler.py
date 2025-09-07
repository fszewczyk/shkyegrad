import time
from contextlib import ContextDecorator

class Profiler(ContextDecorator):
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *exc):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        return False