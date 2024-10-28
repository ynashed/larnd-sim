"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""

import jax.numpy as jnp
from jax.profiler import annotate_function
from jax import jit, vmap, lax, random, debug
from jax.nn import sigmoid
from functools import partial

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("DETSIM MODULE PARAMETERS")

# @annotate_function
# @jit
# def accumulate_signals(wfs, currents_idx, charge, response, pixID, start_ticks):
#     # Get the number of pixels and ticks
#     Npixels, Nticks = wfs.shape

#     # Compute indices for updating wfs, taking into account start_ticks
#     start_indices = jnp.expand_dims(pixID, axis=-1) * Nticks + start_ticks[:, jnp.newaxis, jnp.newaxis]
#     end_indices = start_indices + jnp.arange(response.shape[-1])

#     # Flatten the indices
#     flat_indices = jnp.ravel(end_indices)
#     print("flat_indices", flat_indices.shape)
#     print("start_indices", start_indices.shape)
#     print("end_indices", end_indices.shape)

#     # Update wfs with accumulated signals
#     wfs = wfs.ravel()
#     wfs = wfs.at[(flat_indices,)].add(signals.ravel())
#     return wfs.reshape((Npixels, Nticks))

# @partial(jit, static_argnames='signal_length')
# def accumulate_signals(wfs, currents_idx, charge, response, pixID, start_ticks, signal_length):
#     #pixID: Ncurrents*Nparallel_pixels
#     #currents_idx: (Ncurrents*Nparallel_pixels, 2)
#     # Get the number of pixels and ticks
#     Npixels, Nticks = wfs.shape

#     # Compute indices for updating wfs, taking into account start_ticks
#     start_indices = pixID * Nticks + start_ticks # shape (Ncurrents*Nparallel_pixels)
    

#     end_indices = start_indices[..., None] + jnp.arange(signal_length) # shape (Ncurrents*Nparallel_pixels, signal_length)
#     debug.print("currents_idx:{currents_idx}", currents_idx=currents_idx.shape)

#     # Flatten the indices
#     flat_indices = jnp.ravel(end_indices) # shape (Ncurrents*Nparallel_pixels*signal_length)
#     # print("flat_indices", flat_indices.shape)
#     # print("start_indices", start_indices.shape)
#     # print("end_indices", end_indices.shape)

#     Nx, Ny, Nt = response.shape

#     signal_indices = jnp.ravel((currents_idx[..., 0, None]*Ny + currents_idx[..., 1, None])*Nt + jnp.arange(response.shape[-1] - signal_length, response.shape[-1])) # shape (Ncurrents*Nparallel_pixels, signal_length)
#     print("response.take(signal_indices)", response.take(signal_indices).shape)
#     print("jnp.repeat(charge, signal_length)", jnp.repeat(charge, signal_length).shape)
#     print("flat_indices", flat_indices)

#     # Update wfs with accumulated signals
#     wfs = wfs.ravel()
#     wfs = wfs.at[(flat_indices,)].add(response.take(signal_indices)*jnp.repeat(charge, signal_length))

#     return wfs.reshape((Npixels, Nticks))


@partial(jit, static_argnames='signal_length')
def accumulate_signals(wfs, currents_idx, charge, response, pixID, start_ticks, signal_length):
    # Get the number of pixels and ticks
    Npixels, Nticks = wfs.shape

    # Compute indices for updating wfs, taking into account start_ticks
    time_ticks = start_ticks[..., None] + jnp.arange(signal_length)

    time_ticks = jnp.where((time_ticks <= 0 ) | (time_ticks >= wfs.shape[1] - 1), 0, time_ticks+1) # it should be start_ticks +1 in theory but we cheat by putting the cumsum in the garbage too when strarting at 0 to mimic the expected behavior

    
    start_indices = pixID * Nticks

    end_indices = start_indices[..., None] + time_ticks

    # Flatten the indices
    flat_indices = jnp.ravel(end_indices)

    
    # print("flat_indices", flat_indices.shape)
    # print("start_indices", start_indices.shape)
    # print("end_indices", end_indices.shape)

    Nx, Ny, Nt = response.shape

    signal_indices = jnp.ravel((currents_idx[..., 0, None]*Ny + currents_idx[..., 1, None])*Nt + jnp.arange(response.shape[-1] - signal_length, response.shape[-1]))
    # print("signal_indices", signal_indices.shape)


    # Update wfs with accumulated signals
    wfs = wfs.ravel()
    wfs = wfs.at[(flat_indices,)].add(response.take(signal_indices)*jnp.repeat(charge, signal_length))
    return wfs.reshape((Npixels, Nticks))


@annotate_function
@jit
def pixel2id(params, pixel_x, pixel_y, pixel_plane, eventID):
    """
    Convert the x,y,plane tuple to a unique identifier

    Args:
        pixel_x (int): number of pixel pitches in x-dimension
        pixel_y (int): number of pixel pitches in y-dimension
        pixel_plane (int): pixel plane number

    Returns:
        unique integer id
    """
    # outside = (pixel_x >= params.n_pixels[0]) | (pixel_y >= params.n_pixels[1])
    outside = (pixel_x >= params.n_pixels_x) | (pixel_y >= params.n_pixels_y)
    # return jnp.where(outside, -1, pixel_x + params.n_pixels[0] * (pixel_y + params.n_pixels[1] * (pixel_plane + params.tpc_borders.shape[0]*eventID)))
    return jnp.where(outside, -1, pixel_x + params.n_pixels_x * (pixel_y + params.n_pixels_y * (pixel_plane + params.tpc_borders.shape[0]*eventID)))

# @annotate_function
@jit
def id2pixel(params, pid):
    """
    Convert the unique pixel identifer to an x,y,plane tuple

    Args:
        pid (int): unique pixel identifier
    Returns:
        tuple: number of pixel pitches in x-dimension,
            number of pixel pitches in y-dimension,
            pixel plane number
    """
    # return (pid % params.n_pixels[0], (pid // params.n_pixels[0]) % params.n_pixels[1],
    #         (pid // (params.n_pixels[0] * params.n_pixels[1])) % params.tpc_borders.shape[0],
    #         pid // (params.n_pixels[0] * params.n_pixels[1]*params.tpc_borders.shape[0]))
    return (pid % params.n_pixels_x, (pid // params.n_pixels_x) % params.n_pixels_y,
            (pid // (params.n_pixels_x * params.n_pixels_y)) % params.tpc_borders.shape[0],
            pid // (params.n_pixels_x * params.n_pixels_y*params.tpc_borders.shape[0]))

# @annotate_function
@partial(jit, static_argnames=['fields'])
def generate_electrons(tracks, fields, rngkey=0):
    key = random.PRNGKey(rngkey)
    sigmas = jnp.stack([tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("long_diff")]], axis=1)
    rnd_pos = random.normal(key, (tracks.shape[0], 3))*sigmas
    electrons = tracks.copy()
    electrons = electrons.at[:, fields.index('x')].set(electrons[:, fields.index('x')] + rnd_pos[:, 0])
    electrons = electrons.at[:, fields.index('y')].set(electrons[:, fields.index('y')] + rnd_pos[:, 1])
    electrons = electrons.at[:, fields.index('z')].set(electrons[:, fields.index('z')] + rnd_pos[:, 2])

    return electrons

# @annotate_function
@partial(jit, static_argnames=['fields'])
def get_pixels(params, electrons, fields):
    n_neigh = params.number_pix_neighbors

    borders = lax.map(lambda i: params.tpc_borders[i], electrons[:, fields.index("pixel_plane")].astype(int))
    pos = jnp.stack([(electrons[:, fields.index("x")] - borders[:, 0, 0]) // params.pixel_pitch,
            (electrons[:, fields.index("y")] - borders[:, 1, 0]) // params.pixel_pitch], axis=1)

    pixels = (pos + 0.5).astype(int)

    X, Y = jnp.mgrid[-n_neigh:n_neigh+1, -n_neigh:n_neigh+1]
    shifts = jnp.vstack([X.ravel(), Y.ravel()]).T
    pixels = pixels[:, jnp.newaxis, :] + shifts[jnp.newaxis, :, :]

    # outside = (pixel_x >= params.n_pixels[0]) | (pixel_y >= params.n_pixels[1])
    outside = (pixels[:, :, 0] >= params.n_pixels_x) | (pixels[:, :, 1] >= params.n_pixels_y)
    # return jnp.where(outside, -1, pixel_x + params.n_pixels[0] * (pixel_y + params.n_pixels[1] * (pixel_plane + params.tpc_borders.shape[0]*eventID)))
    return jnp.where(outside, -1, pixels[:, :, 0] + params.n_pixels_x * (pixels[:, :, 1] + params.n_pixels_y * (electrons[:, fields.index("pixel_plane")].astype(int)[:, jnp.newaxis] + params.tpc_borders.shape[0]*electrons[:, fields.index("eventID")].astype(int)[:, jnp.newaxis])))

    # return pixel2id(params, pixels[:, :, 0], pixels[:, :, 1], electrons[:, fields.index("pixel_plane")].astype(int)[:, jnp.newaxis], electrons[:, fields.index("eventID")].astype(int)[:, jnp.newaxis])

@annotate_function
@jit
def truncexpon(x, loc=0, scale=1, y_cutoff=-10., rate=100):
    """
    A truncated exponential distribution.
    To shift and/or scale the distribution use the `loc` and `scale` parameters.
    """
    y = (x - loc) / scale
    # Use smoothed mask to make derivatives nicer
    # y cutoff stops exp from blowing up -- should be far enough away from 0 that sigmoid is small
    y = jnp.maximum(y, y_cutoff)
    return sigmoid(rate*y)*jnp.exp(-y) / scale

@annotate_function
@jit
def current_model(t, t0, x, y):
    """
    Parametrization of the induced current on the pixel, which depends
    on the of arrival at the anode (:math:`t_0`) and on the position
    on the pixel pad.

    Args:
        t (float): time where we evaluate the current
        t0 (float): time of arrival at the anode
        x (float): distance between the point on the pixel and the pixel center
            on the :math:`x` axis
        y (float): distance between the point on the pixel and the pixel center
            on the :math:`y` axis

    Returns:
        float: the induced current at time :math:`t`
    """
    B_params = (1.060, -0.909, -0.909, 5.856, 0.207, 0.207)
    C_params = (0.679, -1.083, -1.083, 8.772, -5.521, -5.521)
    D_params = (2.644, -9.174, -9.174, 13.483, 45.887, 45.887)
    t0_params = (2.948, -2.705, -2.705, 4.825, 20.814, 20.814)

    a = B_params[0] + B_params[1] * x + B_params[2] * y + B_params[3] * x * y + B_params[4] * x * x + B_params[
        5] * y * y
    b = C_params[0] + C_params[1] * x + C_params[2] * y + C_params[3] * x * y + C_params[4] * x * x + C_params[
        5] * y * y
    c = D_params[0] + D_params[1] * x + D_params[2] * y + D_params[3] * x * y + D_params[4] * x * x + D_params[
        5] * y * y
    shifted_t0 = t0 + t0_params[0] + t0_params[1] * x + t0_params[2] * y + \
                    t0_params[3] * x * y + t0_params[4] * x * x + t0_params[5] * y * y

    a = jnp.minimum(a, 1)

    return a * truncexpon(-t, -shifted_t0, b) + (1 - a) * truncexpon(-t, -shifted_t0, c)


@partial(jit, static_argnames=['fields'])
def time_intervals(params, tracks, fields):
    """
    Find the value of the longest signal time and stores the start
    time of each segment.

    Args:
        event_id_map (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array containing
            the event ID corresponding to each track
        tracks (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array containing the segment
            information
        fields (list): an ordered string list of field/column name of the tracks structured array
    Returns:
        track_starts (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array where
            we store the segments start time
        time_max (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array where we store
            the longest signal time
    """
    tracks_t_end = tracks[:, fields.index("t_end")]
    tracks_t_start = tracks[:, fields.index("t_start")]
    t_end = jnp.minimum(jnp.full_like(tracks_t_end, params.time_interval[1]),
                        ((tracks_t_end + params.time_padding + 0.5 / params.vdrift) / params.t_sampling) * params.t_sampling)
    t_start = jnp.maximum(jnp.full_like(tracks_t_start, params.time_interval[0]),
                            ((tracks_t_start - params.time_padding) / params.t_sampling) * params.t_sampling)
    t_length = t_end - t_start

    time_max = jnp.trunc(jnp.max(t_length / params.t_sampling + 1))
    # debug.print("time_max: {time_max}", time_max=t_end)
    return t_start, time_max

@jit
def overlapping_segment(x, y, start, end, radius):
    """
    Calculates the segment of the track defined by start, end that overlaps
    with a circle centered at x,y

    """

    eps = 1e-6

    dx = x - start[:, 0]
    dy = y - start[:, 1]

    vx = end[:, 0] - start[:, 0]
    vy = end[:, 1] - start[:, 1]
    l = jnp.sqrt(vx**2 + vy**2) + eps
    vx = vx/l
    vy = vy/l

    s = (dx * vx + dy * vy)/l # position of point of closest approach

    r = jnp.sqrt((dx - vx * s * l)**2 + (dy - vy * s * l)**2)
    r = jnp.minimum(r, radius)
    s_plus = lax.select(r >= radius, jnp.zeros_like(r), s + jnp.sqrt(radius**2 - r**2) / l)
    s_minus = lax.select(r >= radius, jnp.zeros_like(r), s - jnp.sqrt(radius**2 - r**2) / l)    

    s_plus = jnp.minimum(s_plus, 1)
    s_plus = jnp.maximum(s_plus, 0)
    s_minus = jnp.minimum(s_minus, 1)
    s_minus = jnp.maximum(s_minus, 0)

    new_start = jnp.column_stack((start[:, 0] * (1 - s_minus) + end[:, 0] * s_minus,
                 start[:, 1] * (1 - s_minus) + end[:, 1] * s_minus,
                 start[:, 2] * (1 - s_minus) + end[:, 2] * s_minus))
    new_end = jnp.column_stack((start[:, 0] * (1 - s_plus) + end[:, 0] * s_plus,
               start[:, 1] * (1 - s_plus) + end[:, 1] * s_plus,
               start[:, 2] * (1 - s_plus) + end[:, 2] * s_plus))

    return new_start, new_end

# @annotate_function
@partial(jit, static_argnames=['fields'])
def current_mc(params, electrons, pixels_coord, fields):
    nticks = int(5/params.t_sampling)
    ticks = jnp.linspace(0, 5, nticks).reshape((1, nticks)).repeat(electrons.shape[0], axis=0)#

    x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[..., 0])
    y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[..., 1])
    # signals = jnp.array((electrons.shape[0], ticks.shape[1]))

    z_anode = lax.map(lambda i: params.tpc_borders[i][2][0], electrons[:, fields.index("pixel_plane")].astype(int))

    #TODO: Actually write something consistent here for the time. Needs it to be >0 for now
    t0 = jnp.abs(electrons[:, fields.index('z')] - z_anode) / params.vdrift# - params.time_window

    # t0 = t0 - jnp.min(t0) + 5

    ticks = ticks + t0[:, jnp.newaxis]

    return t0, current_model(ticks, t0[:, jnp.newaxis], x_dist[:, jnp.newaxis], y_dist[:, jnp.newaxis])*electrons[:, fields.index("n_electrons")].reshape((electrons.shape[0], 1))*params.e_charge

# @partial(jit, static_argnames=['fields'])
# def current_lut(params, response, electrons, pixels_coord, fields):
#     x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[..., 0])
#     y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[..., 1])
#     # print("x_dist", x_dist.shape)
#     # print("pixels_coord", pixels_coord.shape)
#     z_anode = lax.map(lambda i: params.tpc_borders[i][2][0], electrons[:, fields.index("pixel_plane")].astype(int))
#     t0 = (jnp.abs(electrons[:, fields.index('z')] - z_anode) / params.vdrift + electrons[:, fields.index('z')] - params.time_padding)
    
#     i = (x_dist/params.response_bin_size).astype(int)
#     j = (y_dist/params.response_bin_size).astype(int)


#     i = jnp.clip(i, 0, response.shape[0] - 1)
#     j = jnp.clip(j, 0, response.shape[1] - 1)

#     # currents = electrons[:, fields.index("n_electrons")][..., None, None]*response[i, j, :]#*params.e_charge
#     currents_idx = jnp.stack([i, j], axis=-1)#*params.e_charge

#     # debug.print("currents_idx -> {currents_idx}", currents_idx=currents_idx)

#     return t0, currents_idx

@partial(jit, static_argnames=['fields'])
def current_lut(params, response, electrons, pixels_coord, fields):
    x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[..., 0])
    y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[..., 1])
    # print("x_dist", x_dist.shape)
    # print("pixels_coord", pixels_coord.shape)
    z_anode = lax.map(lambda i: params.tpc_borders[i][2][0], electrons[:, fields.index("pixel_plane")].astype(int))
    # t0 = (jnp.abs(electrons[:, fields.index('z')] - z_anode) / params.vdrift + electrons[:, fields.index('z')] - params.time_padding)
    t0 = jnp.abs(electrons[:, fields.index('z')] - z_anode) / params.vdrift
    
    i = (x_dist/params.response_bin_size).astype(int)
    j = (y_dist/params.response_bin_size).astype(int)


    i = jnp.clip(i, 0, response.shape[0] - 1)
    j = jnp.clip(j, 0, response.shape[1] - 1)

    # currents = electrons[:, fields.index("n_electrons")][..., None, None]*response[i, j, :]#*params.e_charge
    currents_idx = jnp.stack([i, j], axis=-1)#*params.e_charge

    # debug.print("currents_idx -> {currents_idx}", currents_idx=currents_idx)

    return t0, currents_idx

@partial(jit, static_argnames=['fields'])
def tracks_current(params, pixels, tracks, fields):
    """
    This function calculates the charge induced on the pixels by the input tracks.

    Args:
        pixels (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): 3D array with dimensions S x P x 2, where S is
            the number of track segments, P is the number of pixels and the third dimension
            contains the two pixel ID numbers.
        tracks (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): 2D array containing the detector segments.
        time_max (int) : total number of time ticks (see time_intervals)
        fields (list): an ordered string list of field/column name of the tracks structured array
    Returns:
        signals (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): 3D array with dimensions S x P x T,
            where S is the number of track segments, P is the number of pixels, and T is
            the number of time ticks.
    """
    #INPUT TRACKS SHOULD ALREADY BE SORTED BY Z
    #pixels (tid, pid_x, pid_y, p_x, p_y)
    #TODO: Check if the gradients are ok with this
    # it = jnp.arange(0, params.time_max)
    t_start = jnp.round((tracks[:, fields.index("t_start")] - params.time_padding) / params.t_sampling) * params.t_sampling
    t_end = jnp.round((tracks[:, fields.index("t_end")] + params.time_padding) / params.t_sampling) * params.t_sampling
    # debug.print("t_start: {t_start}", t_start=t_start)
    # debug.print("t_end: {t_end}", t_end=t_end)

    start = jnp.stack([tracks[:, fields.index("x_start")],
                        tracks[:, fields.index("y_start")],
                        tracks[:, fields.index("z_start")]], axis=1)
    end = jnp.stack([tracks[:, fields.index("x_end")],
                    tracks[:, fields.index("y_end")],
                    tracks[:, fields.index("z_end")]], axis=1)

    segment = end - start
    # debug.print("segment= {segment}", segment=segment)
    # length = jnp.linalg.norm(segment, ord=2, axis=1, keepdims=True)
    # length = jnp.sqrt(segment[:, 0]**2 + segment[:, 1]**2 + segment[:, 2]**2)
    length = jnp.sqrt(jnp.sum(segment**2, axis=1, keepdims=True))

    direction = segment / length
    sigmas = jnp.stack([tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("long_diff")]], axis=1)

    # The impact factor is the the size of the transverse diffusion or, if too small,
    # half the diagonal of the pixel pad
    impact_factor = jnp.maximum(jnp.sqrt((5 * sigmas[:, 0]) ** 2 + (5 * sigmas[:, 1]) ** 2),
                                jnp.full_like(sigmas[:, 0], params.pixel_pitch / jnp.sqrt(2))) * 2

    #TODO: need to check the pitch, might be off by a multiple of half the pixel pitch    
    subsegment_start, subsegment_end = lax.stop_gradient(overlapping_segment(pixels[:, 3], pixels[:, 4], start, end, impact_factor))
    subsegment = subsegment_end - subsegment_start
    # debug.print("subsegment= {subsegment}", subsegment=subsegment)

    subsegment_length = lax.stop_gradient(jnp.sqrt(jnp.sum(subsegment**2, axis=1, keepdims=True)))
    # debug.print("subsegment_length= {subsegment_length}", subsegment_length=subsegment_length)
    
    # debug.print("subsegment_length: {subsegment_length}", subsegment_length=subsegment_length)
    # debug.print("subsegment_z_displacement: {subsegment_z_displacement}", subsegment_z_displacement=subsegment_start[:, 2]-start[:, 2])

    key = random.PRNGKey(0)
    keys = random.split(key, tracks.shape[0])
    nstep = lax.stop_gradient(jnp.squeeze(jnp.maximum(jnp.round(subsegment_length / 0.001), 1)).astype(int))
    charge = tracks[:, fields.index("n_electrons")] * jnp.squeeze(subsegment_length/length)/nstep
    z_anode = lax.map(lambda i: params.tpc_borders[i][2][0], tracks[:, fields.index("pixel_plane")].astype(int))
    signals = vmap(lambda *args: current_mc(params, *args))(subsegment_start, subsegment_length, t_start, pixels[:, 3], pixels[:, 4], direction, sigmas, charge, z_anode, nstep, keys)

    return signals
    #TODO: Stop if subsegment_length==0