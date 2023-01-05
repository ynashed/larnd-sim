import torch
from .lineprof import LineProf
from functools import wraps
import atexit

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

def disable_memprof():
    print("Disabling the memory line profiler")
    global_line_profiler.disable()

def enable_file_output():
    global_line_profiler.set_file_output(True)

def enable_data_export():
    global_line_profiler.set_data_export(True)


def memprof(columns = base_columns):
    def decorator(func):
        if global_line_profiler.enabled:
            if not global_line_profiler.has_registered():
                def print_stats_atexit():
                    global_line_profiler.print_stats()
                atexit.register(print_stats_atexit)
            global_line_profiler.add_function(func, columns)
            
        @wraps(func)
        def run_func(*args, **kwargs):
            return func(*args, **kwargs)
        return run_func

    return decorator