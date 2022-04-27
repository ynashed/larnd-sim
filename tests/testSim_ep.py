#!/usr/bin/env python3

from torch.utils.data import DataLoader

from optimize.dataio import TracksDataset
from optimize.utils import all_sim, get_id_map, embed_adc_list

from larndsim.sim_with_grad import sim_with_grad
import torch
import os

def calc_forward():
    track_chunk = 1
    pixel_chunk = 1
    detector_props = "larndsim/detector_properties/module0.yaml" 
    pixel_layouts = "larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml"
    input_file =  "tests/data/test_inputs.h5"
    device = 'cpu'
    

    dataset = TracksDataset(filename=input_file, ntrack=1)
    selected_tracks_torch = dataset[0][0:1]
    track_fields = dataset.get_track_fields()  
 
    sim_target = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=False)
    sim_target.load_detector_properties(detector_props, pixel_layouts)

    event_id_map, unique_eventIDs = get_id_map(selected_tracks_torch, track_fields, device)
    selected_tracks_torch = selected_tracks_torch.to(device)

    target, pix_target = all_sim(sim_target, selected_tracks_torch, track_fields,
                                 event_id_map, unique_eventIDs,
                                 return_unique_pix=True)
    embed_target = embed_adc_list(sim_target, target, pix_target)
  
    return embed_target

def test_forward():
    output_path = 'tests/data/testSim_ep_out.pth'
    outputs = calc_forward()
    if os.path.exists(output_path):
        check = torch.load(output_path)
        assert torch.allclose(outputs, check)
    else:
        print("Saving new comparison file")
        torch.save(outputs, output_path)
