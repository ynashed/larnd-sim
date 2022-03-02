import numpy as np
from numpy.lib import recfunctions as rfn
import torch

def torch_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return torch.from_numpy(tracks_np).float()

def structured_from_torch(tracks_torch, dtype):
    return rfn.unstructured_to_structured(tracks_torch.cpu().numpy(), dtype=dtype)


def batch(index, tracks, size=10, max_seg=-1):
    n_seg = 0
    out_trk = []
    while n_seg < size:
        rand_ev = np.random.choice(list(index.keys()))
        rand_track = np.random.randint(0, len(index[rand_ev]))
        mask = (tracks['eventID']== rand_ev) & (tracks['trackID'] == index[rand_ev][rand_track])
        n_seg += np.sum(mask)
        
        out_trk.append(torch_from_structured(tracks[mask].copy()))
       
    out = torch.cat(out_trk, dim=0)
    if max_seg != -1 and len(out) > max_seg:
        idxs = np.random.permutation(np.arange(max_seg))
        return out[idxs]
    else:
        return out

def get_id_map(selected_tracks, fields, device):
    # Here we build a map between tracks and event IDs (no param dependence, so np should be ok)
    unique_eventIDs = np.unique(selected_tracks[:, fields.index('eventID')])
    event_id_map = np.searchsorted(unique_eventIDs,np.asarray(selected_tracks[:, fields.index('eventID')]))
    event_id_map_torch = torch.from_numpy(event_id_map).to(device)
    
    return event_id_map_torch, unique_eventIDs

# Update parameters for training loop
def update_grad_param(sim, name, value):
    setattr(sim, name, value)
    sim.track_gradients([name])

def param_l2_reg(param, sim):
    sigma = (ranges[param]['up'] - ranges[param]['down'])/2.
    return ((ranges[param]['nom']-getattr(sim, param))**2)/(sigma**2)

def calc_reg_loss(param_list, sim, regs):
    reg_loss = 0.
    for param in param_list:
        reg_loss+=regs[param]*param_l2_reg(param, sim)
        
    return reg_loss
