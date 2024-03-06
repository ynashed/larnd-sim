"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""

import jax.numpy as jnp
from jax.profiler import annotate_function
from jax import grad, jit, vmap, lax, make_jaxpr, random, debug
from jax.nn import sigmoid
from functools import partial

from .consts_ep import consts
from .fee_ep import fee
from .utils import diff_arange

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("DETSIM MODULE PARAMETERS")

@annotate_function
@jit
def accumulate_signals(wfs, signals, pixID, start_ticks):
    # Get the number of pixels and ticks
    Npixels, Nticks = wfs.shape

    # Compute indices for updating wfs, taking into account start_ticks
    start_indices = jnp.expand_dims(pixID, axis=1) * Nticks + start_ticks[:, jnp.newaxis]
    end_indices = start_indices + jnp.arange(signals.shape[1])

    # Flatten the indices
    flat_indices = jnp.ravel(end_indices)

    # Update wfs with accumulated signals
    wfs = wfs.ravel()
    wfs = wfs.at[(flat_indices,)].add(signals.ravel())
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
    outside = (pixel_x >= params.n_pixels[0]) | (pixel_y >= params.n_pixels[1])
    return jnp.where(outside, -1, pixel_x + params.n_pixels[0] * (pixel_y + params.n_pixels[1] * pixel_plane) + eventID*params.n_pixels[0]*params.n_pixels[1]*params.tpc_borders.shape[0])

@annotate_function
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
    return (pid % params.n_pixels[0], (pid // params.n_pixels[0]) % params.n_pixels[1],
            (pid // (params.n_pixels[0] * params.n_pixels[1])) % params.tpc_borders.shape[0],
            pid // (params.n_pixels[0] * params.n_pixels[1]*params.tpc_borders.shape[0]))

@annotate_function
@partial(jit, static_argnames=['fields'])
def generate_electrons(tracks, fields):
    key = random.PRNGKey(0)
    sigmas = jnp.stack([tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("long_diff")]], axis=1)
    rnd_pos = random.normal(key, (tracks.shape[0], 3))*sigmas
    electrons = tracks.copy()
    electrons = electrons.at[:, fields.index('x')].set(electrons[:, fields.index('x')] + rnd_pos[:, 0])
    electrons = electrons.at[:, fields.index('y')].set(electrons[:, fields.index('y')] + rnd_pos[:, 1])
    electrons = electrons.at[:, fields.index('z')].set(electrons[:, fields.index('z')] + rnd_pos[:, 2])

    return electrons

@annotate_function
@partial(jit, static_argnames=['fields'])
def get_pixels(params, electrons, fields):
    borders = lax.map(lambda i: params.tpc_borders[i], electrons[:, fields.index("pixel_plane")].astype(int))
    pos = jnp.stack([(electrons[:, fields.index("x")] - borders[:, 0, 0]) // params.pixel_pitch,
            (electrons[:, fields.index("y")] - borders[:, 1, 0]) // params.pixel_pitch], axis=1)

    pixels = (pos + 0.5).astype(int)
    return pixel2id(params, pixels[:, 0], pixels[:, 1], electrons[:, fields.index("pixel_plane")].astype(int), electrons[:, fields.index("eventID")].astype(int))

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

@annotate_function
@partial(jit, static_argnames=['fields'])
def current_mc(params, electrons, pixels_coord, fields):
    nticks = int(5/params.t_sampling)
    ticks = jnp.linspace(0, 5, nticks).reshape((1, nticks)).repeat(electrons.shape[0], axis=0)#

    x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[:, 0])
    y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[:, 1])

    # signals = jnp.array((electrons.shape[0], ticks.shape[1]))

    z_anode = lax.map(lambda i: params.tpc_borders[i][2][0], electrons[:, fields.index("pixel_plane")].astype(int))

    #TODO: Actually write something consistent here for the time. Needs it to be >0 for now
    t0 = jnp.abs(electrons[:, fields.index('z')] - z_anode) / params.vdrift# - params.time_window
    t0 = t0 - jnp.min(t0) + 5

    ticks = ticks + t0[:, jnp.newaxis]

    return t0, current_model(ticks, t0[:, jnp.newaxis], x_dist[:, jnp.newaxis], y_dist[:, jnp.newaxis])*electrons[:, fields.index("n_electrons")].reshape((electrons.shape[0], 1))*params.e_charge

# @jit
# def current_mc(params, subsegment_start, subsegment_length, t_start, x_p, y_p,
#                direction, sigmas, charge, z_anode, nstep, rng_key):
#     # nstep = max(round(subsegment_length / MIN_STEP_SIZE), 1)
#     step = subsegment_length / nstep # refine step size
#     it = jnp.arange(0, params.time_max)
#     time_ticks = t_start + it * params.t_sampling
#     zero_current = jnp.zeros((params.time_max,))
#     #TODO: Ensure that time_tick is >= 0

#     def current_mc_step(istep, carry):
#         signal, key = carry

#         x = subsegment_start[0] + step * (istep + 0.5) * direction[0]
#         y = subsegment_start[1] + step * (istep + 0.5) * direction[1]
#         z = subsegment_start[2] + step * (istep + 0.5) * direction[2]

#         rnd_pos = random.normal(key, (3,))*sigmas
#         x = x + rnd_pos[0]
#         y = y + rnd_pos[1]
#         z = z + rnd_pos[2]

#         x_dist = abs(x_p - x)
#         y_dist = abs(y_p - y)
        

#         t0 = jnp.abs(z - z_anode) / params.vdrift - params.time_window
#         # debug.print("direction: {direction}", direction=direction)
#         # debug.print("t0: {t0}", t0=t0)
#         # debug.print("time_ticks: {time_ticks}", time_ticks=time_ticks)
#         # if not t0 < time_tick < t0 + detector.TIME_WINDOW:
#         #     continue

#         # if x_dist > detector.RESPONSE_BIN_SIZE * response.shape[0]:
#         #     continue
#         # if y_dist > detector.RESPONSE_BIN_SIZE * response.shape[1]:
#         #     continue

#         signal += charge * lax.select(t0 < time_ticks, current_model(time_ticks, t0, x_dist, y_dist), zero_current)

#         return (signal , random.split(key, 1)[0])
#         # return random.split(key, 1)[0]
    
#     def select_current_mc(istep, carry):
#         return lax.cond(istep < nstep, current_mc_step, lambda i, c: c, istep, carry)
    
#     # total_current = lax.fori_loop(0, nstep, current_mc_step, rng_key)
    
#     # total_current, _ = lax.fori_loop(0, nstep, current_mc_step, (zero_current, rng_key))
#     total_current, _ = lax.fori_loop(0, 100, select_current_mc, (zero_current, rng_key))
#     # print((subsegment_start, subsegment_length, x_p, y_p, direction, sigmas, charge, nstep, rng_key))

#     return total_current


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
    
    # z_poca, z_start, z_end = self.z_interval(start, end, x_p, y_p, impact_factor)

    # z_start_int = z_start - 4 * sigmas[:, 2][...,jnp.newaxis]
    # z_end_int = z_end + 4 * sigmas[:, 2][...,jnp.newaxis]

    # x_start, y_start = self.track_point(start, direction, z_start)
    # x_end, y_end = self.track_point(start, direction, z_end)

    # y_step = (jnp.abs(y_end - y_start) + 8 * sigmas[:, 1][...,jnp.newaxis]) / (self.sampled_points - 1)
    # x_step = (jnp.abs(x_end - x_start) + 8 * sigmas[:, 0][...,jnp.newaxis]) / (self.sampled_points - 1)

    # # This was a // divide, implement?
    # t_start = jnp.maximum(self.time_interval[0],
    #                         (tracks_ep[:, fields.index("t_start")] - self.time_padding)
    #                         / self.t_sampling * self.t_sampling)

    # time_tick = t_start[:, jnp.newaxis] + it * self.t_sampling
    # tpc_borders_ep = jnp.from_numpy(pixels, self.tpc_borders).float32()
    # borders = jnp.stack([tpc_borders_ep[x.astype(int)] for x in tracks_ep[:, fields.index("pixel_plane")]])

    # signals = jnp.zeros(z_start, shape=(pixels.shape[0], pixels.shape[1], time_max.astype(int).item()))
    # for it in range(0, z_start.shape[0], self.track_chunk):
    #     it_end = min(it + self.track_chunk, z_start.shape[0])
    #     if not self.skip_pixels:
    #         pix_end_range = z_start.shape[1]
    #     else: # ASSUMES THAT TRACK_CHUNK = 1
    #         pix_end_range = min(npixels[it], z_start.shape[1])
    #     for ip in range(0, pix_end_range, self.pixel_chunk):
    #         ip_end = min(ip + self.pixel_chunk, pix_end_range)
    #         if tracks_jnp.raw.grad_fn is not None:
    #             # Torch checkpointing needs torch tensors for both input and output
    #             current_sum = checkpoint.checkpoint(self.calc_total_current, 
    #                                                 *(x_start[it:it_end, ip:ip_end].raw, y_start[it:it_end, ip:ip_end].raw, z_start.raw, 
    #                                                 z_end.raw, z_start_int[it:it_end, ip:ip_end].raw, z_end_int[it:it_end, ip:ip_end].raw, z_poca[it:it_end, ip:ip_end].raw, 
    #                                                 x_p[it:it_end, ip:ip_end].raw, y_p[it:it_end, ip:ip_end].raw, x_step[it:it_end, ip:ip_end].raw, y_step[it:it_end, ip:ip_end].raw, borders[it:it_end].raw, direction[it:it_end].raw, 
    #                                                 sigmas[it:it_end].raw,
    #                                                 tracks_ep[it:it_end, fields.index("n_electrons")].raw, start[it:it_end].raw, segment[it:it_end].raw, time_tick[it:it_end].raw, self.vdrift))
    #         else:
    #             current_sum = self.calc_total_current( 
    #                                                 x_start[it:it_end, ip:ip_end].raw, y_start[it:it_end, ip:ip_end].raw, z_start.raw,
    #                                                 z_end.raw, z_start_int[it:it_end, ip:ip_end].raw, z_end_int[it:it_end, ip:ip_end].raw, z_poca[it:it_end, ip:ip_end].raw,
    #                                                 x_p[it:it_end, ip:ip_end].raw, y_p[it:it_end, ip:ip_end].raw, x_step[it:it_end, ip:ip_end].raw, y_step[it:it_end, ip:ip_end].raw, borders[it:it_end].raw, direction[it:it_end].raw,
    #                                                 sigmas[it:it_end].raw,
    #                                                 tracks_ep[it:it_end, fields.index("n_electrons")].raw, start[it:it_end].raw, segment[it:it_end].raw, time_tick[it:it_end].raw, self.vdrift)

    #         signals = jnp.index_update(signals, jnp.index[it:it_end, ip:ip_end, :], jnp.astensor(current_sum))
    
    # return signals.raw

class detsim(consts):
    def __init__(self, track_chunk, pixel_chunk, skip_pixels=False):
        self.track_chunk = track_chunk
        self.pixel_chunk = pixel_chunk
        self.skip_pixels = skip_pixels
        consts.__init__(self)

    def z_interval(self, start_point, end_point, x_p, y_p, tolerance, eps=1e-12):
        """
        Here we calculate the interval in the drift direction for the pixel pID
        using the impact factor

        Args:
            start_point (tuple): coordinates of the segment start
            end_point (tuple): coordinates of the segment end
            x_p (float): pixel center `x` coordinate
            y_p (float): pixel center `y` coordinate
            tolerance (float): maximum distance between the pixel center and
                the segment

        Returns:
            tuple: `z` coordinate of the point of closest approach (POCA),
            `z` coordinate of the first slice, `z` coordinate of the last slice.
            (0,0,0) if POCA > tolerance.
        """
        cond = start_point[:, 0] < end_point[:, 0]
        start = jnp.where(cond[..., jnp.newaxis], start_point, end_point)
        end = jnp.where(cond[..., jnp.newaxis], end_point, start_point)

        xs, ys = start[:, 0], start[:, 1]
        xe, ye = end[:, 0], end[:, 1]

        m = (ye - ys) / (xe - xs + eps)
        q = (xe * ys - xs * ye) / (xe - xs + eps)

        a, b, c = m[...,jnp.newaxis], -1, q[...,jnp.newaxis]

        x_poca = (b * (b * x_p[...,0] - a * y_p[...,0]) - a * c) / (a * a + b * b)
        doca = jnp.abs(a * x_p[...,0] + b * y_p[...,0] + c) / jnp.sqrt(a * a  + b * b)

        vec3D = end - start
        length3D = jnp.norms.l2(vec3D, axis=1, keepdims=True)
        dir3D = vec3D / length3D

        #TODO: Fixme. Not efficient. Should just flip start and end
        end = end[...,jnp.newaxis]
        start = start[..., jnp.newaxis]
        cond2 = x_poca > end[:, 0]
        cond1 = x_poca < start[:, 0]
        doca = jnp.where(cond2,
                        jnp.sqrt((x_p[...,0] - end[:, 0]) ** 2 + (y_p[...,0] - end[:, 1]) ** 2),
                        doca)
        doca = jnp.where(cond1,
                        jnp.sqrt((x_p[...,0] - start[:, 0]) ** 2 + (y_p[...,0] - start[:, 1]) ** 2),
                        doca)

        x_poca = jnp.where(cond2, end[:, 0], x_poca)
        x_poca = jnp.where(cond1, start[:, 0], x_poca)
        z_poca = start[:, 2] + (x_poca - start[:, 0]) / dir3D[:, 0][..., jnp.newaxis] * dir3D[:, 2][..., jnp.newaxis]

        length2D = jnp.norms.l2(vec3D[...,:2], axis=1, keepdims=True)
        dir2D = vec3D[...,:2] / length2D

        #Check this - abs avoids nans
        deltaL2D = jnp.sqrt(jnp.abs(tolerance[..., jnp.newaxis] ** 2 - doca ** 2))  # length along the track in 2D

        x_plusDeltaL = x_poca + deltaL2D * dir2D[:,0][..., jnp.newaxis]  # x coordinates of the tolerance range
        x_minusDeltaL = x_poca - deltaL2D * dir2D[:,0][..., jnp.newaxis]
        plusDeltaL = (x_plusDeltaL - start[:,0,:]) / dir3D[:,0][..., jnp.newaxis]  # length along the track in 3D
        minusDeltaL = (x_minusDeltaL - start[:,0,:]) / dir3D[:,0][..., jnp.newaxis]  # of the tolerance range

        plusDeltaZ = start[:,2,:] + dir3D[:,2][..., jnp.newaxis] * plusDeltaL  # z coordinates of the
        minusDeltaZ = start[:,2,:] + dir3D[:,2][..., jnp.newaxis] * minusDeltaL  # tolerance range

        cond = tolerance[..., jnp.newaxis] > doca
        z_poca = jnp.where(cond, z_poca, 0)
        z_min_delta = jnp.where(cond, jnp.minimum(minusDeltaZ, plusDeltaZ), 0)
        z_max_delta = jnp.where(cond, jnp.maximum(minusDeltaZ, plusDeltaZ), 0)
        return z_poca, z_min_delta, z_max_delta

    def erf_hack(self, input):
        return jnp.astensor(torch.erf(input.raw))

    def rho(self, point, q, start, sigmas, segment):
        """
        Function that returns the amount of charge at a certain point in space

        Args:
            point (tuple): point coordinates
            q (float): total charge
            start (tuple): segment start coordinates
            sigmas (tuple): diffusion coefficients
            segment (tuple): segment sizes

        Returns:
            float: the amount of charge at `point`.
        """
        x, y, z = point
        Deltax, Deltay, Deltaz = segment[..., 0], segment[..., 1], segment[..., 2]
        Deltar = jnp.sqrt(Deltax**2+Deltay**2+Deltaz**2)
        seg_step = segment / Deltar[:, jnp.newaxis]
        sigma2 = sigmas ** 2
        double_sigma2 = 2 * sigma2
        a = ((Deltax/Deltar) * (Deltax/Deltar) / (double_sigma2[:, 0]) + \
             (Deltay/Deltar) * (Deltay/Deltar) / (double_sigma2[:, 1]) + \
             (Deltaz/Deltar) * (Deltaz/Deltar) / (double_sigma2[:, 2]))
        factor = q/Deltar/(sigmas[:, 0]*sigmas[:, 1]*sigmas[:, 2]*sqrt(8*pi*pi*pi))
        sqrt_a_2 = 2*jnp.sqrt(a)

        x_component = (x - start[:, 0, jnp.newaxis, jnp.newaxis])
        y_component = (y - start[:, 1, jnp.newaxis, jnp.newaxis])
        z_component = (z - start[:, 2, jnp.newaxis, jnp.newaxis])

        b = -( (x_component / (sigma2[:, 0] / seg_step[:, 0])[..., jnp.newaxis, jnp.newaxis])[..., jnp.newaxis, jnp.newaxis] +
               (y_component / (sigma2[:, 1] / seg_step[:, 1])[..., jnp.newaxis, jnp.newaxis])[:, :, jnp.newaxis, :, jnp.newaxis] +
               (z_component / (sigma2[:, 2] / seg_step[:, 2])[..., jnp.newaxis, jnp.newaxis])[:, :, jnp.newaxis, jnp.newaxis, :] )

        delta = (x_component**2/(double_sigma2[:, 0, jnp.newaxis, jnp.newaxis]))[..., jnp.newaxis, jnp.newaxis] + \
                (y_component**2/(double_sigma2[:, 1, jnp.newaxis, jnp.newaxis]))[:, :, jnp.newaxis, :, jnp.newaxis] + \
                (z_component**2/(double_sigma2[:, 2, jnp.newaxis, jnp.newaxis]))[:, :, jnp.newaxis, jnp.newaxis, :]
        padded_sqrt_a_2 = sqrt_a_2[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        integral = sqrt(pi) * \
                   (-self.erf_hack(b/padded_sqrt_a_2) +
                    self.erf_hack((b + 2*(a*Deltar)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis])/
                                  padded_sqrt_a_2)) / padded_sqrt_a_2
        
        # expo = jnp.exp(b*b/(4*a[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis]) - delta + jnp.log(factor[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis]) + jnp.log(integral))
        # expo = jnp.where(expo.isnan(), 0, expo)
        #Avoid logs by bringing down - should be equiv?
        expo_factor = jnp.where(jnp.logical_and(factor[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis] != 0, integral != 0), b*b/(4*a[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis]) - delta, 0)
        expo = jnp.exp(expo_factor)*factor[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis]*integral
        # logger.debug(f"Got {np.count_nonzero(np.isnan(expo.raw.detach().cpu().numpy()))} NaNs in expo")
        expo = jnp.where(expo.isnan(), 0, expo)

        
        
        #TODO: Figure out a way to do the sum over the sampling cube here.
        # Ask about the x_dist, y_dist > pixel_pitch/2 conditions in the original simulation
        return expo




    def track_point(self, start, direction, z):
        """
        This function returns the segment coordinates for a point along the `z` coordinate

        Args:
            start (tuple): start coordinates
            direction (tuple): direction coordinates
            z (float): `z` coordinate corresponding to the `x`, `y` coordinates

        Returns:
            tuple: the (x,y) pair of coordinates for the segment at `z`
        """
        l = (z - start[:, 2][...,jnp.newaxis]) / direction[:, 2][...,jnp.newaxis]
        xl = start[:, 0][...,jnp.newaxis] + l * direction[:, 0][...,jnp.newaxis]
        yl = start[:, 1][...,jnp.newaxis] + l * direction[:, 1][...,jnp.newaxis]

        return xl, yl

    def get_pixel_coordinates(self, pixels):
        """
        Returns the coordinates of the pixel center given the pixel IDs
        """
        tpc_borders_ep = jnp.from_numpy(pixels, self.tpc_borders).float32()
        plane_id = pixels[..., 0] // self.n_pixels[0]
        borders = jnp.stack([tpc_borders_ep[x.astype(int)] for x in plane_id])

        pix_x = (pixels[..., 0] - self.n_pixels[0] * plane_id) * self.pixel_pitch + borders[..., 0, 0]
        pix_y = pixels[..., 1] * self.pixel_pitch + borders[..., 1, 0]
        return pix_x[...,jnp.newaxis], pix_y[...,jnp.newaxis]

    def calc_total_current(self, x_start, y_start, z_start,
                           z_end, z_start_int, z_end_int, z_poca, 
                           x_p, y_p, x_step, y_step, borders, direction, sigmas, tracks_ep, start, segment, time_tick, vdrift):

        x_start = jnp.astensor(x_start)
        y_start = jnp.astensor(y_start)
        z_start = jnp.astensor(z_start)
        z_end = jnp.astensor(z_end)
        z_start_int = jnp.astensor(z_start_int)
        z_end_int = jnp.astensor(z_end_int)
        z_poca = jnp.astensor(z_poca)
        x_p = jnp.astensor(x_p)
        y_p = jnp.astensor(y_p)
        x_step = jnp.astensor(x_step)
        y_step = jnp.astensor(y_step)
        borders = jnp.astensor(borders)
        direction = jnp.astensor(direction)
        sigmas = jnp.astensor(sigmas)
        tracks_ep = jnp.astensor(tracks_ep)
        start = jnp.astensor(start)
        segment = jnp.astensor(segment)
        time_tick = jnp.astensor(time_tick)

        z_sampling = self.t_sampling / 2.
        z_steps = jnp.maximum(self.sampled_points, ((jnp.abs(z_end_int
                                                    - z_start_int) / z_sampling)+1).astype(int))

        z_step = (z_end_int - z_start_int) / (z_steps - 1)

        iz = jnp.arange(z_steps, 0, z_steps.max().item())
        z =  z_start_int[:, :, jnp.newaxis] + iz[jnp.newaxis, jnp.newaxis, :] * z_step[..., jnp.newaxis]

        t0 = (jnp.abs(z - borders[:, 2, 0, jnp.newaxis, jnp.newaxis]) - 0.5) / vdrift

        # FIXME: this sampling is far from ideal, we should sample around the track
        # and not in a cube containing the track
        ix = jnp.arange(iz, 0, self.sampled_points)
        x = x_start[:, :, jnp.newaxis] + \
            jnp.sign(direction[:, 0, jnp.newaxis, jnp.newaxis]) *\
            (ix[jnp.newaxis, jnp.newaxis, :] *
             x_step[:, :, jnp.newaxis]  - 4 * sigmas[:, 0, jnp.newaxis, jnp.newaxis])
        x_dist = jnp.abs(x_p - x)

        iy = jnp.arange(iz, 0, self.sampled_points)
        y = y_start[:, :, jnp.newaxis] + \
            jnp.sign(direction[:, 1, jnp.newaxis, jnp.newaxis]) * \
            (iy[jnp.newaxis, jnp.newaxis, :] *
             y_step[:, :, jnp.newaxis] - 4 * sigmas[:, 1, jnp.newaxis, jnp.newaxis])
        y_dist = jnp.abs(y_p - y)

        charge = self.rho((x, y, z),
                          tracks_ep,
                          start, sigmas, segment) *\
                 jnp.abs(x_step[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]) *\
                 jnp.abs(y_step[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]) *\
                 jnp.abs(z_step[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis])

        # mask z_poca rows
        charge *= (z_poca != 0)[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]

        # mask x rows
        charge *= (x_dist < self.pixel_pitch / 2)[..., jnp.newaxis, jnp.newaxis]

        # mask y elements
        charge *= (y_dist < self.pixel_pitch / 2)[:, :, jnp.newaxis, :, jnp.newaxis]

        x_dist = jnp.minimum(x_dist, self.pixel_pitch / 2)
        y_dist = jnp.minimum(y_dist, self.pixel_pitch / 2)
        current_out = self.current_model(time_tick[:, jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis],
                                         t0[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
                                         x_dist[:, :, jnp.newaxis, :, jnp.newaxis, jnp.newaxis],
                                         y_dist[:, :, jnp.newaxis, jnp.newaxis, :, jnp.newaxis])


        total_current = charge[:, :, jnp.newaxis, ...] * current_out * self.e_charge
        
        return total_current.sum(axis=(3, 4, 5)).raw


    def sum_pixel_signals(self, pixels_signals, signals, track_starts, index_map):
        """
        This function sums the induced current signals on the same pixel.

        Args:
            pixels_signals (:obj:`numpy.ndarray`): 2D array that will contain the
                summed signal for each pixel. First dimension is the pixel ID, second
                dimension is the time tick
            signals (:obj:`numpy.ndarray`): 3D array with dimensions S x P x T,
                where S is the number of track segments, P is the number of pixels, and T is
                the number of time ticks.
            track_starts (:obj:`numpy.ndarray`): 1D array containing the starting time of
                each track
            index_map (:obj:`numpy.ndarray`): 2D array containing the correspondence between
                the track index and the pixel ID index.
        """

        signals = jnp.astensor(signals)
        track_starts = jnp.astensor(track_starts)
        index_map = jnp.astensor(index_map)

        # Set up index map to match with signal shape
        index = index_map[..., jnp.newaxis]

        # Set up time map to match with signal shape. To implement: jnp.round
        itime = ((track_starts / self.t_sampling + 0.5)[:, jnp.newaxis, jnp.newaxis] +
                 jnp.arange(signals, 0, signals.shape[2])[jnp.newaxis, jnp.newaxis, :])
 
        # Each signal index now has a corresponding pixel/time index
        exp_index = jnp.tile(index, (1,1,signals.shape[2]))
        exp_itime = jnp.tile(itime, (1, signals.shape[1], 1))

        # Put pixel/time/signal together and flatten
        idxs = jnp.stack((exp_index, exp_itime, signals), axis=-1)
        flat_idxs = idxs.reshape((-1, 3))

        # Get unique indices (return_inverse doesn't exist for ep)
        unique_idxs, idx_inv = flat_idxs[:, :2].astype(int).raw.unique(dim=0, return_inverse=True)

        unique_idxs = jnp.astensor(unique_idxs)
        idx_inv = jnp.astensor(idx_inv)
        # Sum over values for unique indices - scatter_add_ doesn't exist in jnp. Can loop, but slow, e.g.
        #out = jnp.zeros(signals, shape=(len(unique_idxs)))
        #for i in range(flat_idxs.shape[0]):
         #   out = out.index_update(idx_inv[i], out[idx_inv[i]]+flat_idxs[i, 2])
        res = jnp.astensor(jnp.zeros(signals, shape=(len(unique_idxs))).raw.scatter_add_(0, idx_inv.raw, flat_idxs[:, 2].raw))

        output = jnp.index_update(jnp.astensor(pixels_signals), (unique_idxs[:,0].astype(int),
                                                               unique_idxs[:,1].astype(int)), res)

        return output.raw



    # def backtrack_adcs(self, tracks, adc_list, adc_times_list, track_pixel_map, event_id_map, unique_evids, backtracked_id,
    #                    shift):
    #     pedestal = floor((fee.V_PEDESTAL - fee.V_CM) * fee.ADC_COUNTS / (fee.V_REF - fee.V_CM))
    #
    #     ip = cuda.grid(1)
    #
    #     if ip < adc_list.shape[0]:
    #         for itrk in range(track_pixel_map.shape[1]):
    #             track_index = track_pixel_map[ip][itrk]
    #             if track_index >= 0:
    #                 track_start_t = tracks["t_start"][track_index]
    #                 track_end_t = tracks["t_end"][track_index]
    #                 evid = unique_evids[event_id_map[track_index]]
    #                 for iadc in range(adc_list[ip].shape[0]):
    #
    #                     if adc_list[ip][iadc] > pedestal:
    #                         adc_time = adc_times_list[ip][iadc]
    #                         evid_time = adc_time // (time_interval[1] * 3)
    #
    #                         if track_start_t - self.time_padding < adc_time - evid_time * time_interval[
    #                             1] * 3 < track_end_t + consts.time_padding + 0.5 / self.vdrift:
    #                             counter = 0
    #
    #                             while counter < backtracked_id.shape[2] and backtracked_id[ip, iadc, counter] != -1:
    #                                 counter += 1
    #
    #                             if counter < backtracked_id.shape[2]:
    #                                 backtracked_id[ip, iadc, counter] = track_index + shift
    #
    #
    # def get_track_pixel_map(self, track_pixel_map, unique_pix, pixels):
    #     # index of unique_pix array
    #     index = cuda.grid(1)
    #
    #     upix = unique_pix[index]
    #
    #     for itr in range(pixels.shape[0]):
    #         for ipix in range(pixels.shape[1]):
    #             pID = pixels[itr][ipix]
    #             if upix[0] == pID[0] and upix[1] == pID[1]:
    #                 imap = 0
    #                 while imap < track_pixel_map.shape[1] and track_pixel_map[index][imap] != -1:
    #                     imap += 1
    #                 if imap < track_pixel_map.shape[1]:
    #                     track_pixel_map[index][imap] = itr
