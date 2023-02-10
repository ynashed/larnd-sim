import torch
from .profiling import memprof, disable_memprof, enable_file_output, enable_data_export, global_line_profiler

@memprof(columns=
    ('active_bytes.all.peak',
    'reserved_bytes.all.peak',
    'active_bytes.all.current'))
def test_func():
    A = torch.ones(1000, 1000).cuda()
    B = torch.ones(1000, 1000).cuda()

    C = A + B
    for i in range(3):
        global_line_profiler.add_note({'iter': i})
        D = inner_func(C)
        global_line_profiler.checkpoint()

@memprof(columns=
    ('active_bytes.all.peak',
    'reserved_bytes.all.peak',
    'active_bytes.all.current'))
def inner_func(tensor):
    tensor = torch.exp(tensor)
    A = torch.ones(10000, 10000).cuda()
    return tensor

if __name__ == '__main__':
    # disable_memprof()
    enable_file_output()
    enable_data_export()
    test_func()