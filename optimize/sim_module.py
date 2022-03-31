import os, sys

larndsim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, larndsim_dir)
from larndsim.sim_with_grad import sim_with_grad
from ranges import ranges
from torch import nn
import torch
import numpy as np
from contextlib import ExitStack


class SimModule(nn.Module):
    def __init__(self, track_chunk, pixel_chunk,
                 detector_props, pixel_layouts):
        super(SimModule, self).__init__()
        # Simulation object for target
        self.sim_object = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk)
        self.sim_object.load_detector_properties(detector_props, pixel_layouts)
        self.requires_grad = False

    def set_param(self, param, param_value):
        setattr(self.sim_object, param, param_value)

    def init_params(self, params, history=None):
        for param in params:
            if history is not None:
                self.set_param(param, history[param][-1])
            else:
                self.set_param(param, getattr(self.sim_object, param)/ranges[param]['nom'])

    def track_params(self, params):
        self.sim_object.track_gradients(params)
        self.requires_grad = True

    def get_params(self, params):
        return [getattr(self.sim_object, param) for param in params]

    def forward(self, tracks, fields, event_id_map, unique_eventIDs, return_unique_pix=False):
        with ExitStack() as stack:
            if not self.requires_grad:
                stack.enter_context(torch.no_grad())
        tracks_quench = self.sim_object.quench(tracks, self.sim_object.birks, fields=fields)
        tracks_drift = self.sim_object.drift(tracks_quench, fields=fields)

        active_pixels_torch, neighboring_pixels_torch, n_pixels_list_ep = self.sim_object.get_pixels(tracks_drift,
                                                                                                     fields=fields)

        track_starts_torch, max_length_torch = self.sim_object.time_intervals(event_id_map,
                                                                              tracks_drift,
                                                                              fields=fields)

        signals_ep = self.sim_object.tracks_current(neighboring_pixels_torch, tracks_drift,
                                                    max_length_torch,
                                                    fields=fields)

        unique_pix_torch = torch.empty((0, 2), device=neighboring_pixels_torch.device)
        pixels_signals_torch = torch.zeros((len(unique_pix_torch), len(self.sim_object.time_ticks) * 50),
                                           device=unique_pix_torch.device)

        shapes_torch = neighboring_pixels_torch.shape
        joined_torch = neighboring_pixels_torch.reshape(shapes_torch[0] * shapes_torch[1], 2)

        this_unique_pix_torch = torch.unique(joined_torch, dim=0)
        this_unique_pix_torch = this_unique_pix_torch[
                                (this_unique_pix_torch[:, 0] != -1) & (this_unique_pix_torch[:, 1] != -1), :]
        unique_pix_torch = torch.cat((unique_pix_torch, this_unique_pix_torch), dim=0)

        this_pixels_signals_torch = torch.zeros((len(this_unique_pix_torch), len(self.sim_object.time_ticks) * 50),
                                                device=unique_pix_torch.device)
        pixels_signals_torch = torch.cat((pixels_signals_torch, this_pixels_signals_torch), dim=0)

        pixel_index_map_torch = torch.full((tracks.shape[0], neighboring_pixels_torch.shape[1]), -1,
                                           device=unique_pix_torch.device)
        compare_torch = (neighboring_pixels_torch[..., np.newaxis, :] == unique_pix_torch)

        indices_torch = torch.where(torch.logical_and(compare_torch[..., 0], compare_torch[..., 1]))
        pixel_index_map_torch[indices_torch[0], indices_torch[1]] = indices_torch[2]

        pixels_signals_torch = self.sim_object.sum_pixel_signals(pixels_signals_torch,
                                                                 signals_ep,
                                                                 track_starts_torch,
                                                                 pixel_index_map_torch)

        time_ticks_torch = torch.linspace(0, len(unique_eventIDs) * self.sim_object.time_interval[1] * 3,
                                          pixels_signals_torch.shape[1] + 1)

        integral_list_torch, adc_ticks_list_torch = self.sim_object.get_adc_values(pixels_signals_torch,
                                                                                   time_ticks_torch,
                                                                                   0)
        adc_list_torch = self.sim_object.digitize(integral_list_torch)

        if return_unique_pix:
            return adc_list_torch, unique_pix_torch,
        else:
            return adc_list_torch

    # ADC counts given as list of pixels. Better for loss to embed this in the "full" pixel space
    def embed_adc_list(self, adc_list, unique_pix):
        zero_val = self.sim_object.digitize(torch.tensor(0)).item()
        new_list = torch.ones((self.sim_object.n_pixels[0], self.sim_object.n_pixels[1], adc_list.shape[1]),
                              device=unique_pix.device) * zero_val

        plane_id = unique_pix[..., 0] // self.sim_object.n_pixels[0]
        unique_pix[..., 0] = unique_pix[..., 0] - self.sim_object.n_pixels[0] * plane_id

        new_list[unique_pix[:, 0].long(), unique_pix[:, 1].long(), :] = adc_list

        return new_list
