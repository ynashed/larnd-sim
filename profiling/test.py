import torch
from .profiling import memprof


@memprof(columns=('active_bytes.all.peak', 'reserved_bytes.all.peak'))
def test_func():
    A = torch.ones(1000, 1000).cuda()
    B = A = torch.ones(1000, 1000).cuda()

    C = A + B

    D = inner_func(C)
    E = inner_func(C)

@memprof()
def inner_func(tensor):
    tensor = torch.exp(tensor)
    return tensor

if __name__ == '__main__':
    test_func()