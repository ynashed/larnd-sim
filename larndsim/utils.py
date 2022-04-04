import eagerpy as ep

def diff_linspace(start, end, num, endpoint=False):
    start = ep.astensor(start)
    end = ep.astensor(end)
    num = int(num)
    
    delta = (end - start) / num
    if endpoint:
        delta *= num / (num - 1)
    return start + ep.arange(end, 0, num)*delta

def diff_arange(start, end):
    return ep.flip(ep.stack([end-j-1. for j in range((end-start).astype(int).item())]), axis=0)
