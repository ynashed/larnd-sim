"""
Module that finds which pixels lie on the projection on the anode plane
of each track segment. It can eventually include also the neighboring
pixels.
"""

import eagerpy as ep
import numpy as np
from math import ceil

from .consts_ep import consts

import logging

logging.basicConfig()
logger = logging.getLogger('pixels_from_track')
logger.setLevel(logging.WARNING)
logger.info("PIXEL_FROM_TRACK MODULE PARAMETERS")

class pixels_from_track(consts):
    def __init__(self):
        super().__init__()

    def get_pixels(self, tracks, fields):
        """
        For all tracks, takes the xy start and end position
        and calculates all impacted pixels by the track segment

        Args:
            tracks (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array where we store the
                track segments information
            fields (list): an ordered string list of field/column name of the tracks structured array
        Returns:
            active_pixels (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array where we store
                the IDs of the pixels directly below the projection of the segments
            neighboring_pixels (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array where we store
                the IDs of the pixels directly below the projection of
                the segments and the ones next to them
            n_pixels_list (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): number of total involved
                pixels
        """
        tracks_ep = ep.astensor(tracks)
        tpc_borders_ep = ep.from_numpy(tracks_ep, self.tpc_borders).float32()
        borders = ep.stack([tpc_borders_ep[x.astype(int)] for x in tracks_ep[:, fields.index("pixel_plane")]])
        start_pixel = ep.stack([(tracks_ep[:, fields.index("x_start")] - borders[:, 0, 0]) // self.pixel_pitch +
                                self.n_pixels[0] * tracks_ep[:, fields.index("pixel_plane")],
                                (tracks_ep[:, fields.index("y_start")] - borders[:, 1, 0]) // self.pixel_pitch], axis=1)
        end_pixel = ep.stack([(tracks_ep[:, fields.index("x_end")] - borders[:, 0, 0]) // self.pixel_pitch +
                              self.n_pixels[0] * tracks_ep[:, fields.index("pixel_plane")],
                              (tracks_ep[:, fields.index("y_end")] - borders[:, 1, 0]) // self.pixel_pitch], axis=1)

        longest_pix = ceil(ep.max(tracks_ep[:, fields.index("dx")]).item() / self.pixel_pitch)
        max_radius = ceil(ep.max(tracks_ep[:, fields.index("tran_diff")]).item() * 5 / self.pixel_pitch)

        max_pixels = int((longest_pix * 4 + 6) * max_radius * 1.5)
        max_active_pixels = int(longest_pix * 1.5)
        active_pixels = self.get_active_pixels(start_pixel, end_pixel, max_active_pixels)

        neighboring_pixels, n_pixels_list = self.get_neighboring_pixels(active_pixels, max_radius + 1, max_pixels)
        return active_pixels.raw, neighboring_pixels.raw, n_pixels_list


    def _bresenhamline_nslope(self, slope, eps=1e-12):
        """
        Normalize slope for Bresenham's line algorithm.
        """
        scale = ep.max(ep.abs(slope), axis=1)[..., ep.newaxis]
        normalized_slope = slope / (scale + eps)
        return normalized_slope, scale


    def get_active_pixels(self, start, end, max_pixels):
        """
        Converts track segement to an array of active pixels
        using Bresenham algorithm used to convert line to grid.

        Args:
            start (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): (n_pts x 2) x, y coordinates of the start pixel
            end (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): (n_pts x 2) x, y coordinates of the end pixel
            max_pixels (int): maximum length of returned lines
        Returns:
            tot_pixels (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): (n_pts x max_pixels, 2) array where we store
                the IDs of the pixels directly below the projection of
                the segments
        """

        nslope, scale = self._bresenhamline_nslope(end - start)
        indices = ep.arange(nslope, 0, max_pixels)
        step = ep.stack([indices, indices], axis=1)
        tot_pixels = start[:, ep.newaxis, :] + nslope[:, ep.newaxis, :] * step
        tot_pixels = (tot_pixels + 0.5).astype(int)
        tot_pixels = ep.where(ep.tile(step, [tot_pixels.shape[0], 1]).reshape(tot_pixels.shape) > scale[..., ep.newaxis],
                              -1, tot_pixels)
        # TODO: check if plane_id is important
        return tot_pixels


    def get_neighboring_pixels(self, active_pixels, radius, max_pixels):
        """
         For each active_pixel, it includes all
         neighboring pixels within a specified radius

         Args:
            active_pixels (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array where we store
                 the IDs of the pixels directly below the projection of
                 the segments
            radius (int): number of layers of neighboring pixels we
                 want to consider
            max_pixels (int): maximum length of returned array
         Returns:
            neighboring_pixels (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`) array where we store the IDs of the
            pixels directly below the projection of the segments and the ones next to them
            n_pixels_list (list): number of total involved pixels
         """
        r = ep.arange(active_pixels, -radius, radius + 1)
        X, Y = ep.meshgrid(r, r, indexing='ij')
        neighbor_indices = ep.stack((X, Y), axis=-1)
        neighbor_indices = ep.reshape(neighbor_indices, [-1, 2])

        neighboring_pixels = []
        n_pixels_list = []
        for ti, track in enumerate(active_pixels):
            track = track[track >= 0].reshape([-1, 2])
            n_indices = ep.tile(neighbor_indices, [track.shape[0], 1]) + \
                        ep.tile(track, [1, neighbor_indices.shape[0]]).reshape([-1, 2])
            n_indices = n_indices[:, 0] * self.n_pixels[1] + n_indices[:, 1]
            n_indices_idx =np.unique(n_indices.raw.numpy(), return_index=True)[1]
            n_indices = n_indices[np.sort(n_indices_idx)]
            n_indices = ep.stack([n_indices // self.n_pixels[1],
                                  n_indices % self.n_pixels[1]], axis=1)
            n_pixels_list.append(int(n_indices.shape[0]))
            neighboring_pixels.append(ep.pad(n_indices, ((0, max_pixels - n_indices.shape[0]), (0, 0)),
                                             mode='constant', value=-1))
        

        neighboring_pixels = ep.stack(neighboring_pixels)

        plane_ids = neighboring_pixels[...,0] // self.n_pixels[0]
        cond = (plane_ids < self.tpc_borders.shape[0])
        neighboring_pixels = ep.where(cond[..., ep.newaxis], neighboring_pixels, -1)
       
        # TODO: check if plane_id is important
        return neighboring_pixels, n_pixels_list
