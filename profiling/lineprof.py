import inspect
import sys
import warnings
import torch
import time

from .records import Records

from dataclasses import dataclass
@dataclass
class Func:
    lines: list
    start_line: int
    __qualname__: str

class LineProf:
    def __init__(self, *functions):
        self._code_infos = {}
        self._raw_line_records = []
        self.enabled = False
        self.registered = {}
        self.file_output = False
        self.export_data = False
        for func in functions:
            self.add_function(func)

    def add_function(self, func, columns):
        try:
            code_hash = hash(func.__code__)
        except AttributeError:
            warnings.warn(
                f"Could not extract a code object for the object {func}")
            return
        if code_hash not in self._code_infos:
            first_line = inspect.getsourcelines(func)[1]
            self._code_infos[code_hash] = {
                'func': func,
                'first_line': first_line,
                'prev_line': first_line,
                'prev_record': -1,
            }
            self.registered[code_hash] = columns

        if self.enabled:
            self.register_callback()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def register_callback(self):
        if self._code_infos:
            sys.settrace(self._trace_callback)

    def _reset_cuda_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def enable(self):
        if not torch.cuda.is_available():
            print('No CUDA device could be found')
            sys.exit(0)
        torch.cuda.empty_cache()
        self._reset_cuda_stats()
        self.enabled = True
        self.register_callback()

    def disable(self):
        self.enabled = False
        sys.settrace(None)

    def set_file_output(self, val):
        self.file_output = bool(val)

    def set_data_export(self, val):
        self.export_data = bool(val) 

    def clear(self):
        self._code_infos = {}
        self._raw_line_records = []

    def has_registered(self):
        return bool(self.registered)

    def export(self, fname):
        # Pickle does not want to pickle the function objects that are decorated... so we need to save only part of the info
        import pickle
        import inspect


        simplified_code_infos = {h : {'func': Func(*inspect.getsourcelines(func['func']), func['func'].__qualname__)} for h, func in self._code_infos.items()}

        to_export = {
            'line_records': self._raw_line_records,
            'code_infos': simplified_code_infos
        }


        with open(fname, 'wb') as f:
            pickle.dump(to_export, f)


    def _trace_callback(self, frame, event, *_):
        if event == 'call':
            return self._trace_callback

        code_hash = hash(frame.f_code)
        if event in ['line', 'return'] and code_hash in self._code_infos:
            code_info = self._code_infos[code_hash]
            self._raw_line_records.append({
                'code_hash': code_hash,
                'line': code_info['prev_line'],
                'prev_record_idx': code_info['prev_record'],
                'time': time.time_ns(),
                **torch.cuda.memory_stats()})

            self._reset_cuda_stats()

            if event == 'line':
                code_info['prev_line'] = frame.f_lineno
                code_info['prev_record'] = len(self._raw_line_records)-1
            elif event == 'return':
                code_info['prev_line'] = code_info['first_line']
                code_info['prev_record'] = -1

    def print_stats(self):
        if not self.enabled or not self._raw_line_records or not self._code_infos:
            return

        records = Records(self._raw_line_records, self._code_infos)
        print(f"Starting to analyze the memory records ; got {len(self._raw_line_records)} records for {len(self.registered)} functions to analyze\n")

        ofilename = None
        curtime = time.strftime('%Y%m%d-%H%M%S')
        if self.file_output:
            ofilename = f"memprof_{curtime}.txt"
            print(f"Writing memprof output in {ofilename}")
        if self.export_data:
            fname = f"memprof_{curtime}.pkl"
            print(f"Writing memprof raw data in {fname}")
            self.export(fname)

        for code_hash, columns in self.registered.items():
            func = self._code_infos[code_hash]['func']
            ostring = str(records.display(func, columns))
            sys.stdout.write(ostring)
            if ofilename:
                with open(ofilename, 'a') as f:
                    f.write(ostring)

