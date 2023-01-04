import torch
from .lineprof import LineProf
from functools import wraps

def to_profile(function):
    @wraps(function)
    def prof(*args, **kwargs):
        with torch.autograd.profiler.record_function(f"func::{function.__name__}"):
            return function(*args, **kwargs)
    
    return prof

global_line_profiler = LineProf()
global_line_profiler.enable()

base_columns = ('active_bytes.all.peak', 'reserved_bytes.all.peak')


def clear_global_line_profiler():
    global_line_profiler.clear()


def memprof(columns = base_columns, enable = True):
    def decorator(func):
        if enable:
            global_line_profiler.add_function(func)
            import atexit

            def print_stats_atexit():
                global_line_profiler.print_stats(func, columns)
            atexit.register(print_stats_atexit)

        @wraps(func)
        def run_func(*args, **kwargs):
            return func(*args, **kwargs)
        return run_func

    return decorator