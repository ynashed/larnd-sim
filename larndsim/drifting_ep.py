"""
Module to implement the propagation of the
electrons towards the anode.
"""

import eagerpy as ep
from . import consts
from .consts import tpc_borders

import logging

logging.basicConfig()
logger = logging.getLogger("drifting")
logger.setLevel(logging.WARNING)
logger.info("DRIFTING MODULE PARAMETERS")


def drift(tracks, fields):
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
        fields (:obj: `string list`) a list of field/column names of the tracks structured array
    """
    tracks_ep = ep.astensor(tracks)
    # TODO: Figure out what to do with pixel plane and tpc borders
    pixel_plane = 0

    z_anode = tpc_borders[pixel_plane][2][0]
    drift_distance = ep.abs(tracks_ep[:, fields.index("z")] - z_anode) - 0.5
    drift_start = ep.abs(ep.minimum(tracks_ep[:, fields.index("z_start")],
                                    tracks_ep[:, fields.index("z_end")]) - z_anode) - 0.5
    drift_end = ep.abs(ep.maximum(tracks_ep[:, fields.index("z_start")],
                                  tracks_ep[:, fields.index("z_end")]) - z_anode) - 0.5
    drift_time = drift_distance / consts.vdrift
    lifetime_red = ep.exp(-drift_time / consts.lifetime)

    tracks[:, fields.index("n_electrons")] = (tracks_ep[:, fields.index("n_electrons")] * lifetime_red).raw

    tracks[:, fields.index("long_diff")] = ep.sqrt((drift_time + 0.5 / consts.vdrift) * 2 * consts.long_diff).raw
    tracks[:, fields.index("tran_diff")] = ep.sqrt((drift_time + 0.5 / consts.vdrift) * 2 * consts.tran_diff).raw
    tracks[:, fields.index("t")] = (tracks_ep[:, fields.index("t")] + drift_time).raw
    tracks[:, fields.index("t_start")] = (tracks_ep[:, fields.index("t_start")] +
                                          ep.minimum(drift_start, drift_end) / consts.vdrift).raw
    tracks[:, fields.index("t_end")] = (tracks_ep[:, fields.index("t_end")] +
                                        ep.maximum(drift_start, drift_end) / consts.vdrift).raw
