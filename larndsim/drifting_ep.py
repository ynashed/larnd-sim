"""
Module to implement the propagation of the
electrons towards the anode.
"""

import eagerpy as ep
import numpy as np

from .consts_ep import consts
# from profiling.profiling import memprof

import logging

logging.basicConfig()
logger = logging.getLogger("drifting")
logger.setLevel(logging.WARNING)
logger.info("DRIFTING MODULE PARAMETERS")


class drift(consts):
    def __init__(self):
        consts.__init__(self)


    # @memprof()
    def drift(self, tracks, fields):
        """
        This function takes as input an array of track segments and calculates
        the properties of the segments at the anode:

          * z coordinate at the anode
          * number of electrons taking into account electron lifetime
          * longitudinal diffusion
          * transverse diffusion
          * time of arrival at the anode

        Args:
            tracks (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array containing the tracks segment information
            fields (list): an ordered string list of field/column name of the tracks structured array
        """
        tracks_ep = ep.astensor(tracks)
        tpc_borders_ep = ep.from_numpy(tracks_ep, self.tpc_borders).float32()
        zMin = ep.minimum(tpc_borders_ep[:, 2, 1] - 2e-2, tpc_borders_ep[:, 2, 0] - 2e-2)
        zMax = ep.maximum(tpc_borders_ep[:, 2, 1] + 2e-2, tpc_borders_ep[:, 2, 0] + 2e-2)

        cond = tracks_ep[:, fields.index("x")][..., ep.newaxis] >= tpc_borders_ep[:, 0, 0][ep.newaxis, ...] - 2e-2
        cond = ep.logical_and(
            tracks_ep[:, fields.index("x")][..., ep.newaxis] <= tpc_borders_ep[:, 0, 1][ep.newaxis, ...] + 2e-2, cond)
        cond = ep.logical_and(
            tracks_ep[:, fields.index("y")][..., ep.newaxis] >= tpc_borders_ep[:, 1, 0][ep.newaxis, ...] - 2e-2, cond)
        cond = ep.logical_and(
            tracks_ep[:, fields.index("y")][..., ep.newaxis] <= tpc_borders_ep[:, 1, 1][ep.newaxis, ...] + 2e-2, cond)
        cond = ep.logical_and(tracks_ep[:, fields.index("z")][..., ep.newaxis] >= zMin[np.newaxis, ...], cond)
        cond = ep.logical_and(tracks_ep[:, fields.index("z")][..., ep.newaxis] <= zMax[np.newaxis, ...], cond)
        pixel_plane = cond.astype(int).argmax(axis=-1)
        tracks_ep = ep.index_update(tracks_ep, ep.index[:, fields.index("pixel_plane")], pixel_plane)
        mask = cond.sum(axis=-1) >= 1

        z_anode = ep.stack([tpc_borders_ep[i][2][0] for i in pixel_plane])
        drift_distance = ep.abs(tracks_ep[:, fields.index("z")] - z_anode) - 0.5
        drift_start = ep.abs(ep.minimum(tracks_ep[:, fields.index("z_start")],
                                        tracks_ep[:, fields.index("z_end")]) - z_anode) - 0.5
        drift_end = ep.abs(ep.maximum(tracks_ep[:, fields.index("z_start")],
                                      tracks_ep[:, fields.index("z_end")]) - z_anode) - 0.5
        drift_time = drift_distance / self.vdrift
        lifetime_red = ep.exp(-drift_time / self.lifetime)

        #TODO: investigate using ep.where instead of masking all values
        #TODO: n_electrons should be int, but truncation makes gradients vanish
        tracks_ep = ep.index_update(tracks_ep, ep.index[:, fields.index("n_electrons")],
                                    (tracks_ep[:, fields.index("n_electrons")] * lifetime_red * mask))
        tracks_ep = ep.index_update(tracks_ep, ep.index[:, fields.index("long_diff")],
                                    ep.sqrt((drift_time + 0.5 / self.vdrift) * 2 * self.long_diff * mask))
        tracks_ep = ep.index_update(tracks_ep, ep.index[:, fields.index("tran_diff")],
                                    ep.sqrt((drift_time + 0.5 / self.vdrift) * 2 * self.tran_diff * mask))
        tracks_ep = ep.index_update(tracks_ep, ep.index[:, fields.index("t")],
                                    (tracks_ep[:, fields.index("t")] + drift_time * mask))
        tracks_ep = ep.index_update(tracks_ep, ep.index[:, fields.index("t_start")],
                                    (tracks_ep[:, fields.index("t_start")] +
                                     ((ep.minimum(drift_start, drift_end) / self.vdrift) * mask)))
        tracks_ep = ep.index_update(tracks_ep, ep.index[:, fields.index("t_end")],
                                    (tracks_ep[:, fields.index("t_end")] +
                                     ((ep.maximum(drift_start, drift_end) / self.vdrift) * mask)))

        return tracks_ep.raw
