"""
Module that finds which pixels lie on the projection on the anode plane
of each track segment. It can eventually include also the neighboring
pixels.
"""

import jax.numpy as jnp
from jax import grad, jit, vmap, lax, make_jaxpr
from jax.experimental import sparse
from functools import partial
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("PIXEL_FROM_TRACK MODULE PARAMETERS")

@jit
def bresenhamline_nslope(slope, eps=1e-12):
    """
    Normalize slope for Bresenham's line algorithm.
    """
    scale = jnp.max(jnp.abs(slope), axis=1)[..., None]
    normalized_slope = slope / (scale + eps)
    return normalized_slope, scale

@partial(jit, static_argnames=['fields'])
def get_active_pixels(params, tracks, fields):
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

    borders = lax.map(lambda i: params.tpc_borders[i], tracks[:, fields.index("pixel_plane")].astype(int))
    start = jnp.stack([(tracks[:, fields.index("x_start")] - borders[:, 0, 0]) // params.pixel_pitch
            + params.n_pixels[0] * tracks[:, fields.index("pixel_plane")],
            (tracks[:, fields.index("y_start")] - borders[:, 1, 0]) // params.pixel_pitch], axis=1)
    end = jnp.stack([(tracks[:, fields.index("x_end")] - borders[:, 0, 0]) // params.pixel_pitch
            + params.n_pixels[0] * tracks[:, fields.index("pixel_plane")],
            (tracks[:, fields.index("y_end")] - borders[:, 1, 0]) // params.pixel_pitch], axis=1)

    nslope, scale = bresenhamline_nslope(end - start)
    indices = jnp.arange(0, params.max_active_pixels)
    step = jnp.stack([indices, indices], axis=1)

    tot_pixels = start[:, jnp.newaxis, :] + nslope[:, jnp.newaxis, :]*step
    tot_pixels = (tot_pixels + 0.5).astype(int)
    tot_pixels = jnp.where(jnp.tile(step, [tot_pixels.shape[0], 1]).reshape(tot_pixels.shape) > scale[..., jnp.newaxis],
                            -1, tot_pixels)
    # TODO: check if plane_id is important
    return tot_pixels

@jit
def get_max_radius(params):
    return jnp.ceil(jnp.sqrt((params.drift_length/params.vdrift + 0.5 / params.vdrift) * 2 * params.tran_diff)).astype(int)

@partial(jit, static_argnames=['fields'])
def get_max_active_pixels(params, tracks, fields):
    longest_pix = jnp.ceil(jnp.max(tracks[:, fields.index("dx")]) / params.pixel_pitch)
    max_active_pixels = (longest_pix * 1.5).astype(int)
    return max_active_pixels

@jit
def get_pixel_coordinates(params, xpitch, ypitch, plane):
    """
    Returns the coordinates of the pixel center given the pixel IDs
    """

    borders = jnp.stack(lax.map(lambda i: params.tpc_borders[i], plane.astype(int)))

    pix_x = xpitch  * params.pixel_pitch + borders[..., 0, 0] + params.pixel_pitch/2
    pix_y = ypitch * params.pixel_pitch + borders[..., 1, 0] + params.pixel_pitch/2
    # return pix_x[...,jnp.newaxis], pix_y[...,jnp.newaxis]
    return jnp.stack([pix_x, pix_y], axis=-1)
    #TODO: REALLY LOOK IN DETAILS AT THE PIXEL LAYOUT THING

# @jit
# def get_pixel_coordinates(params, pixels_sp):
#     """
#     Returns the coordinates of the pixel center given the pixel IDs
#     """

#     plane_id = pixels_sp[..., 1] // params.n_pixels[0]
#     borders = jnp.stack(lax.map(lambda i: params.tpc_borders[i], plane_id.astype(int)))

#     pix_x = (pixels_sp[..., 1] - params.n_pixels[0] * plane_id) * params.pixel_pitch + borders[..., 0, 0] + params.pixel_pitch/2
#     pix_y = pixels_sp[..., 2] * params.pixel_pitch + borders[..., 1, 0] + params.pixel_pitch/2
#     # return pix_x[...,jnp.newaxis], pix_y[...,jnp.newaxis]
#     return jnp.column_stack([pix_x, pix_y])

def pixels_to_sp(active_pixels):
    max_idx = jnp.max(active_pixels)

    ntracks = active_pixels.shape[0]
    npix = active_pixels.shape[1]

    indexing = jnp.column_stack((jnp.repeat(jnp.arange(0, ntracks), npix), jnp.reshape(active_pixels, (ntracks*npix, 2),)))
 
    indexing = indexing[indexing[:, 2] != -1]

    return sparse.BCOO((jnp.ones(indexing.shape[0]), indexing), shape=(ntracks, max_idx, max_idx))

def get_neighboring_pixels_sp(params, pixels_sp):
    r = jnp.arange(-params.max_radius, params.max_radius + 1)
    X, Y = jnp.meshgrid(r, r, indexing='ij')
    variations = jnp.swapaxes(jnp.stack([jnp.zeros_like(X), X, Y]), 0, -1).reshape((2*params.max_radius + 1)**2, 3)
    indices = pixels_sp.indices
    all_indices = jnp.apply_along_axis(lambda x: x + variations, -1, indices)
    all_indices = jnp.reshape(all_indices, (indices.shape[0]*(2*params.max_radius + 1)**2, 3))
    return jnp.unique(all_indices, axis=0)
