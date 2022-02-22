import os, sys

larndsim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, larndsim_dir)
from larndsim.sim_with_grad import sim_with_grad
from torch import nn
import torch
import numpy as np


class SimModule(nn.Module):
    def __init__(self, track_chunk, pixel_chunk,
                 detector_props, pixel_layouts):

        # Simulation object for target
        self.sim_object = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk)
        self.sim_object.load_detector_properties(detector_props, pixel_layouts)

    def forward(self, tracks, fields, event_id_map, unique_eventIDs, return_unique_pix=False):
        #TODO: use this function instead of util.all_sim
        pass
