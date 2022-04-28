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

def all_sim(sim, selected_tracks, fields, event_id_map, unique_eventIDs, return_unique_pix=False):
    selected_tracks_quench = sim.quench(selected_tracks, sim.birks, fields=fields)
    selected_tracks_drift = sim.drift(selected_tracks_quench, fields=fields)

    active_pixels_torch, neighboring_pixels_torch, n_pixels_list_ep = sim.get_pixels(selected_tracks_drift,
                                                                                     fields=fields)

    track_starts_torch, max_length_torch = sim.time_intervals(event_id_map, 
                                                              selected_tracks_drift, 
                                                              fields=fields)
    
    signals_ep = sim.tracks_current(neighboring_pixels_torch, selected_tracks_drift, 
                                          max_length_torch,
                                          fields=fields)

    unique_pix_torch = torch.empty((0, 2), device=neighboring_pixels_torch.device)
    pixels_signals_torch = torch.zeros((len(unique_pix_torch), len(sim.time_ticks)*3),
                                       device=unique_pix_torch.device, dtype=selected_tracks.dtype)

    shapes_torch = neighboring_pixels_torch.shape
    joined_torch = neighboring_pixels_torch.reshape(shapes_torch[0]*shapes_torch[1], 2)

    this_unique_pix_torch = torch.unique(joined_torch, dim=0)
    this_unique_pix_torch = this_unique_pix_torch[(this_unique_pix_torch[:,0] != -1) & (this_unique_pix_torch[:,1] != -1),:]
    unique_pix_torch = torch.cat((unique_pix_torch, this_unique_pix_torch),dim=0)

    this_pixels_signals_torch = torch.zeros((len(this_unique_pix_torch), len(sim.time_ticks)*3),
                                            device=unique_pix_torch.device)
    pixels_signals_torch = torch.cat((pixels_signals_torch, this_pixels_signals_torch), dim=0)

    pixel_index_map_torch = torch.full((selected_tracks.shape[0], neighboring_pixels_torch.shape[1]), -1,
                                       device=unique_pix_torch.device)
    compare_torch = (neighboring_pixels_torch[..., np.newaxis, :] == unique_pix_torch)

    indices_torch = torch.where(torch.logical_and(compare_torch[..., 0], compare_torch[...,1]))
    pixel_index_map_torch[indices_torch[0], indices_torch[1]] = indices_torch[2]
    
    pixels_signals_torch = sim.sum_pixel_signals(pixels_signals_torch,
                                                 signals_ep,
                                                track_starts_torch,
                                                pixel_index_map_torch)
    
    time_ticks_torch = torch.linspace(0, len(unique_eventIDs)*sim.time_interval[1]*3, pixels_signals_torch.shape[1]+1)

    integral_list_torch, adc_ticks_list_torch = sim.get_adc_values(pixels_signals_torch,
                                                                   time_ticks_torch,
                                                                   0)
    adc_list_torch = sim.digitize(integral_list_torch)

    if return_unique_pix:
        return adc_list_torch, unique_pix_torch, adc_list_torch
    else:
        return adc_list_torch

# Update parameters for training loop
def update_grad_param(sim, name, value):
    setattr(sim, name, value)
    sim.track_gradients([name])

# Make tuples for sparse "point cloud" comparison
def embed_adc_list(sim, adc_list, unique_pix, ticks_list):
    zero_val = sim.digitize(torch.tensor(0.))
    mask = (adc_list > zero_val)
    adc_nz = adc_list[mask]
    exp_pix = torch.tile(unique_pix[:, :, None], (1, 1, sim.MAX_ADC_VALUES))
    pix_nz_x = exp_pix[:, 0][mask]
    pix_nz_y = exp_pix[:, 1][mask]
    ticks_list_nz =  ticks_list[mask]
    z_nz =  ticks_list_nz*sim.vdrift
    
    return torch.stack([pix_nz_x, pix_nz_y, z_nz, ticks_list_nz, adc_nz])

# Idea for sparse loss -- compare only non-zero values of guess and target
# Comparison gets min L2 distance on (x,y,t,q) across points in guess for each 
# target point (best match), then takes the mean across target points for the loss.
# If guess == target, min L2 is 0 for all, so loss is 0
def calc_loss(embed_out, embed_targ, return_components = False):
    # Unroll embedding
    pix_out_nz_x, pix_out_nz_y, z_out_nz, ticks_list_out_nz, adc_out_nz = embed_out
    pix_targ_nz_x, pix_targ_nz_y, z_targ_nz, ticks_list_targ_nz, adc_targ_nz = embed_targ

    # Indices for all pairs
    I, J = torch.meshgrid(torch.arange(len(ticks_list_targ_nz)), 
                          torch.arange(len(ticks_list_out_nz)))
    
    # Normalize by mean values to avoid dimension imbalance
    norm_x = ((pix_targ_nz_x.mean() + pix_out_nz_x.mean()) / 2.)**2
    norm_y = ((pix_targ_nz_y.mean() + pix_out_nz_y.mean()) / 2.)**2
    norm_ticks = ((ticks_list_targ_nz.mean() + ticks_list_out_nz.mean()) / 2.)**2
    norm_adc = ((adc_targ_nz.mean() + adc_out_nz.mean()) / 2.)**2

    # Individual component losses (z included to help point matching)
    pix_loss_x = (pix_targ_nz_x[I] - pix_out_nz_x[J])**2
    pix_loss_y = (pix_targ_nz_y[I] - pix_out_nz_y[J])**2
    ticks_loss = (ticks_list_targ_nz[I] - ticks_list_out_nz[J])**2
    z_loss = (z_targ_nz[I] - z_out_nz[J])**2
    adc_loss = (adc_targ_nz[I]-adc_out_nz[J])**2

    # Spatially match pairs
    space_match_idxs = torch.argmin(pix_loss_x + pix_loss_y + z_loss, dim=0)
    #min_idxs = (torch.arange(pix_loss_x.shape[1]), space_match_idxs)
    min_idxs = (space_match_idxs, torch.arange(pix_loss_x.shape[1]))
    # Can return separate components for debugging, otherwise return loss as discussed above
    if return_components:
        return (pix_loss_x, 
                pix_loss_y, 
                ticks_loss, 
                adc_loss)
    else:
        return torch.mean((pix_loss_x/norm_x + pix_loss_y/norm_y + ticks_loss/norm_ticks + adc_loss/norm_adc)[min_idxs])

def param_l2_reg(param, sim):
    sigma = (ranges[param]['up'] - ranges[param]['down'])/2.
    return ((ranges[param]['nom']-getattr(sim, param))**2)/(sigma**2)

def calc_reg_loss(param_list, sim, regs):
    reg_loss = 0.
    for param in param_list:
        reg_loss+=regs[param]*param_l2_reg(param, sim)
        
    return reg_loss
