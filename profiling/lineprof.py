import inspect
import sys
import warnings
import torch
import time

from .records import Records

class LineProf:
    def __init__(self, *functions):
        self._code_infos = {}
        self._raw_line_records = []
        self.enabled = False
        self.registered = {}
        self.file_output = False
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

    def clear(self):
        self._code_infos = {}
        self._raw_line_records = []

    def has_registered(self):
        return bool(self.registered)

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
        if self.file_output:
            ofilename = f"memprof_{time.strftime('%Y%m%d-%H%M%S')}.txt"
            print(f"Writing memprof output in {ofilename}")

        for code_hash, columns in self.registered.items():
            func = self._code_infos[code_hash]['func']
            ostring = str(records.display(func, columns))
            sys.stdout.write(ostring)
            if ofilename:
                with open(ofilename, 'a') as f:
                    f.write(ostring)

