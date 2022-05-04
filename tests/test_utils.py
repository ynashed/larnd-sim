#!/usr/bin/env python3

from torch.utils.data import DataLoader

from optimize.dataio import TracksDataset
from optimize.utils import all_sim, get_id_map, embed_adc_list

from larndsim.sim_with_grad import sim_with_grad

import torch

def calc_forward(with_grad=False, param_list=[], shift=0.05, device='cpu'):
    track_chunk = 1
    pixel_chunk = 2
    detector_props = "larndsim/detector_properties/module0.yaml"
    pixel_layouts = "larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml"
    input_file =  "tests/data/test_inputs.h5"

    dataset = TracksDataset(filename=input_file, ntrack=1)
    selected_tracks_torch = dataset[0][0:1]
    track_fields = dataset.get_track_fields()

    sim_target = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=False)
    sim_target.load_detector_properties(detector_props, pixel_layouts)

    if with_grad:
        for param in param_list:
            setattr(sim_target, param, torch.tensor(getattr(sim_target, param)*(1+shift), requires_grad=True))

    event_id_map, unique_eventIDs = get_id_map(selected_tracks_torch, track_fields, device)
    selected_tracks_torch = selected_tracks_torch.to(device)

    target, pix_target = all_sim(sim_target, selected_tracks_torch.double(), track_fields,
                                 event_id_map, unique_eventIDs,
                                 return_unique_pix=True)
    embed_target = embed_adc_list(sim_target, target, pix_target)

    if with_grad:
        return embed_target, sim_target
    else:
        return embed_target

def loss_fn(guess, targets):
    return torch.nn.MSELoss()(guess, targets)

