"""
Module to implement the propagation of the
electrons towards the anode.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("DRIFTING MODULE PARAMETERS")

@partial(jit, static_argnames='fields')
def drift(params, tracks, fields):
    zMin = jnp.minimum(params.tpc_borders[:, 2, 1] - 2e-2, params.tpc_borders[:, 2, 0] - 2e-2)
    zMax = jnp.maximum(params.tpc_borders[:, 2, 1] + 2e-2, params.tpc_borders[:, 2, 0] + 2e-2)

    cond = tracks[:, fields.index("x")][..., None] >= params.tpc_borders[:, 0, 0][None, ...] - 2e-2
    cond = jnp.logical_and(tracks[:, fields.index("x")][..., None] <= params.tpc_borders[:, 0, 1][None, ...] + 2e-2, cond)
    cond = jnp.logical_and(tracks[:, fields.index("y")][..., None] >= params.tpc_borders[:, 1, 0][None, ...] - 2e-2, cond)
    cond = jnp.logical_and(tracks[:, fields.index("y")][..., None] <= params.tpc_borders[:, 1, 1][None, ...] + 2e-2, cond)
    cond = jnp.logical_and(tracks[:, fields.index("z")][..., None] >= zMin[None, ...], cond)
    cond = jnp.logical_and(tracks[:, fields.index("z")][..., None] <= zMax[None, ...], cond)

    mask = cond.sum(axis=-1) >= 1
    pixel_plane = cond.astype(int).argmax(axis=-1)
    eps = 1e-6
    z_anode = jnp.take(params.tpc_borders, pixel_plane.astype(int), axis=0)[..., 2, 0]
    drift_distance = jnp.abs(tracks[:, fields.index("z")] - z_anode) + eps #Adding alittle something so that it is never equal to zero for grads
    drift_start = jnp.abs(jnp.minimum(tracks[:, fields.index("z_start")],
                                    tracks[:, fields.index("z_end")]) - z_anode)
    drift_end = jnp.abs(jnp.maximum(tracks[:, fields.index("z_start")],
                                  tracks[:, fields.index("z_end")]) - z_anode)
    
    tracks = tracks.at[:, fields.index("pixel_plane")].set(pixel_plane)

    drift_time = drift_distance / params.vdrift
    lifetime_red = jnp.exp(-drift_time / params.lifetime)

    #TODO: investigate using jnp.where instead of masking all values
    tracks = tracks.at[:, fields.index("n_electrons")].set(
        tracks[:, fields.index("n_electrons")] * lifetime_red * mask)
    tracks = tracks.at[:, fields.index("long_diff")].set(
        jnp.sqrt((drift_time + 0.5 / params.vdrift) * 2 * params.long_diff))
    tracks = tracks.at[:, fields.index("tran_diff")].set(
        jnp.sqrt((drift_time + 0.5 / params.vdrift) * 2 * params.tran_diff))
    tracks = tracks.at[:, fields.index("t")].set(
        tracks[:, fields.index("t")] + drift_time * mask + tracks[:, fields.index("t0")])
    tracks = tracks.at[:, fields.index("t_start")].set(
        tracks[:, fields.index("t_start")] + ((jnp.minimum(drift_start, drift_end) / params.vdrift) * mask) + tracks[:, fields.index("t0")])
    tracks = tracks.at[:, fields.index("t_end")].set(
        tracks[:, fields.index("t_end")] + ((jnp.maximum(drift_start, drift_end) / params.vdrift) * mask) + tracks[:, fields.index("t0")])

    return tracks