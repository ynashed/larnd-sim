"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""
import eagerpy as ep
import torch
import numpy as np
from math import pi, ceil, sqrt, erf, exp, log, floor
from torch.utils import checkpoint

from .consts_ep import consts
from .fee_ep import fee
from .utils import diff_arange

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("DETSIM MODULE PARAMETERS")

class detsim(consts):
    def __init__(self, track_chunk, pixel_chunk, skip_pixels=False):
        self.track_chunk = track_chunk
        self.pixel_chunk = pixel_chunk
        self.skip_pixels = skip_pixels
        consts.__init__(self)


    def time_intervals(self, tracks, fields):
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
        tracks_ep = ep.astensor(tracks)
        tracks_t_end = tracks_ep[:, fields.index("t_end")]
        tracks_t_start = tracks_ep[:, fields.index("t_start")]
        t_end = ep.minimum(ep.full_like(tracks_t_end, self.time_interval[1]),
                           ((tracks_t_end + self.time_padding + 0.5 / self.vdrift) / self.t_sampling) * self.t_sampling)
        t_start = ep.maximum(ep.full_like(tracks_t_start, self.time_interval[0]),
                             ((tracks_t_start - self.time_padding) / self.t_sampling) * self.t_sampling)
        t_length = t_end - t_start
        track_starts = t_start.raw

        time_max = (ep.max(t_length / self.t_sampling + 1))
        return track_starts, time_max.raw

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
        start = ep.where(cond[..., ep.newaxis], start_point, end_point)
        end = ep.where(cond[..., ep.newaxis], end_point, start_point)

        xs, ys = start[:, 0], start[:, 1]
        xe, ye = end[:, 0], end[:, 1]

        m = (ye - ys) / (xe - xs + eps)
        q = (xe * ys - xs * ye) / (xe - xs + eps)

        a, b, c = m[...,ep.newaxis], -1, q[...,ep.newaxis]

        x_poca = (b * (b * x_p[...,0] - a * y_p[...,0]) - a * c) / (a * a + b * b)
        doca = ep.abs(a * x_p[...,0] + b * y_p[...,0] + c) / ep.sqrt(a * a  + b * b)

        vec3D = end - start
        length3D = ep.norms.l2(vec3D, axis=1, keepdims=True)
        dir3D = vec3D / length3D

        #TODO: Fixme. Not efficient. Should just flip start and end
        end = end[...,ep.newaxis]
        start = start[..., ep.newaxis]
        cond2 = x_poca > end[:, 0]
        cond1 = x_poca < start[:, 0]
        doca = ep.where(cond2,
                        ep.sqrt((x_p[...,0] - end[:, 0]) ** 2 + (y_p[...,0] - end[:, 1]) ** 2),
                        doca)
        doca = ep.where(cond1,
                        ep.sqrt((x_p[...,0] - start[:, 0]) ** 2 + (y_p[...,0] - start[:, 1]) ** 2),
                        doca)

        x_poca = ep.where(cond2, end[:, 0], x_poca)
        x_poca = ep.where(cond1, start[:, 0], x_poca)
        z_poca = start[:, 2] + (x_poca - start[:, 0]) / dir3D[:, 0][..., ep.newaxis] * dir3D[:, 2][..., ep.newaxis]

        length2D = ep.norms.l2(vec3D[...,:2], axis=1, keepdims=True)
        dir2D = vec3D[...,:2] / length2D

        #Check this - abs avoids nans
        deltaL2D = ep.sqrt(ep.abs(tolerance[..., ep.newaxis] ** 2 - doca ** 2))  # length along the track in 2D

        x_plusDeltaL = x_poca + deltaL2D * dir2D[:,0][..., ep.newaxis]  # x coordinates of the tolerance range
        x_minusDeltaL = x_poca - deltaL2D * dir2D[:,0][..., ep.newaxis]
        plusDeltaL = (x_plusDeltaL - start[:,0,:]) / dir3D[:,0][..., ep.newaxis]  # length along the track in 3D
        minusDeltaL = (x_minusDeltaL - start[:,0,:]) / dir3D[:,0][..., ep.newaxis]  # of the tolerance range

        plusDeltaZ = start[:,2,:] + dir3D[:,2][..., ep.newaxis] * plusDeltaL  # z coordinates of the
        minusDeltaZ = start[:,2,:] + dir3D[:,2][..., ep.newaxis] * minusDeltaL  # tolerance range

        cond = tolerance[..., ep.newaxis] > doca
        z_poca = ep.where(cond, z_poca, 0)
        z_min_delta = ep.where(cond, ep.minimum(minusDeltaZ, plusDeltaZ), 0)
        z_max_delta = ep.where(cond, ep.maximum(minusDeltaZ, plusDeltaZ), 0)
        return z_poca, z_min_delta, z_max_delta

    def erf_hack(self, input):
        return ep.astensor(torch.erf(input.raw))

    def sigmoid(self, x):
        # Using torch here, otherwise we get overflow -- might be able to get around with a trick
        return ep.astensor(torch.sigmoid(x.raw))

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
        Deltar = ep.sqrt(Deltax**2+Deltay**2+Deltaz**2)
        seg_step = segment / Deltar[:, ep.newaxis]
        sigma2 = sigmas ** 2
        double_sigma2 = 2 * sigma2
        a = ((Deltax/Deltar) * (Deltax/Deltar) / (double_sigma2[:, 0]) + \
             (Deltay/Deltar) * (Deltay/Deltar) / (double_sigma2[:, 1]) + \
             (Deltaz/Deltar) * (Deltaz/Deltar) / (double_sigma2[:, 2]))
        factor = q/Deltar/(sigmas[:, 0]*sigmas[:, 1]*sigmas[:, 2]*sqrt(8*pi*pi*pi))
        sqrt_a_2 = 2*ep.sqrt(a)

        x_component = (x - start[:, 0, ep.newaxis, ep.newaxis])
        y_component = (y - start[:, 1, ep.newaxis, ep.newaxis])
        z_component = (z - start[:, 2, ep.newaxis, ep.newaxis])

        b = -( (x_component / (sigma2[:, 0] / seg_step[:, 0])[..., ep.newaxis, ep.newaxis])[..., ep.newaxis, ep.newaxis] +
               (y_component / (sigma2[:, 1] / seg_step[:, 1])[..., ep.newaxis, ep.newaxis])[:, :, ep.newaxis, :, ep.newaxis] +
               (z_component / (sigma2[:, 2] / seg_step[:, 2])[..., ep.newaxis, ep.newaxis])[:, :, ep.newaxis, ep.newaxis, :] )

        delta = (x_component**2/(double_sigma2[:, 0, ep.newaxis, ep.newaxis]))[..., ep.newaxis, ep.newaxis] + \
                (y_component**2/(double_sigma2[:, 1, ep.newaxis, ep.newaxis]))[:, :, ep.newaxis, :, ep.newaxis] + \
                (z_component**2/(double_sigma2[:, 2, ep.newaxis, ep.newaxis]))[:, :, ep.newaxis, ep.newaxis, :]
        padded_sqrt_a_2 = sqrt_a_2[:, ep.newaxis, ep.newaxis, ep.newaxis, ep.newaxis]
        integral = sqrt(pi) * \
                   (-self.erf_hack(b/padded_sqrt_a_2) +
                    self.erf_hack((b + 2*(a*Deltar)[:, ep.newaxis, ep.newaxis, ep.newaxis, ep.newaxis])/
                                  padded_sqrt_a_2)) / padded_sqrt_a_2
        
        # expo = ep.exp(b*b/(4*a[:, ep.newaxis, ep.newaxis, ep.newaxis, ep.newaxis]) - delta + ep.log(factor[:, ep.newaxis, ep.newaxis, ep.newaxis, ep.newaxis]) + ep.log(integral))
        # expo = ep.where(expo.isnan(), 0, expo)
        #Avoid logs by bringing down - should be equiv?
        expo_factor = ep.where(ep.logical_and(factor[:, ep.newaxis, ep.newaxis, ep.newaxis, ep.newaxis] != 0, integral != 0), b*b/(4*a[:, ep.newaxis, ep.newaxis, ep.newaxis, ep.newaxis]) - delta, 0)
        expo = ep.exp(expo_factor)*factor[:, ep.newaxis, ep.newaxis, ep.newaxis, ep.newaxis]*integral
        # logger.debug(f"Got {np.count_nonzero(np.isnan(expo.raw.detach().cpu().numpy()))} NaNs in expo")
        expo = ep.where(expo.isnan(), 0, expo)

        
        
        #TODO: Figure out a way to do the sum over the sampling cube here.
        # Ask about the x_dist, y_dist > pixel_pitch/2 conditions in the original simulation
        return expo

    def truncexpon(self, x, loc=0, scale=1, y_cutoff=-10., rate=100):
        """
        A truncated exponential distribution.
        To shift and/or scale the distribution use the `loc` and `scale` parameters.
        """
        y = (x - loc) / scale
         
        if self.smooth:
            # Use smoothed mask to make derivatives nicer
            # y cutoff stops exp from blowing up -- should be far enough away from 0 that sigmoid is small
            y = ep.maximum(y, y_cutoff)
            return self.sigmoid(rate*y)*ep.exp(-y) / scale
        else:
            # Make everything positive and then mask to avoid nans
            return (ep.exp(-ep.abs(y)) * (y>0)) / scale

    def current_model(self, t, t0, x, y):
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

        a = ep.minimum(a, 1)

        return a * self.truncexpon(-t, -shifted_t0, b) + (1 - a) * self.truncexpon(-t, -shifted_t0, c) 


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
        l = (z - start[:, 2][...,ep.newaxis]) / direction[:, 2][...,ep.newaxis]
        xl = start[:, 0][...,ep.newaxis] + l * direction[:, 0][...,ep.newaxis]
        yl = start[:, 1][...,ep.newaxis] + l * direction[:, 1][...,ep.newaxis]

        return xl, yl

    def get_pixel_coordinates(self, pixels):
        """
        Returns the coordinates of the pixel center given the pixel IDs
        """
        tpc_borders_ep = ep.from_numpy(pixels, self.tpc_borders).float32()
        plane_id = pixels[..., 0] // self.n_pixels[0]
        borders = ep.stack([tpc_borders_ep[x.astype(int)] for x in plane_id])

        pix_x = (pixels[..., 0] - self.n_pixels[0] * plane_id) * self.pixel_pitch + borders[..., 0, 0]
        pix_y = pixels[..., 1] * self.pixel_pitch + borders[..., 1, 0]
        return pix_x[...,ep.newaxis], pix_y[...,ep.newaxis]

    def calc_total_current(self, x_start, y_start, z_start,
                           z_end, z_start_int, z_end_int, z_poca, 
                           x_p, y_p, x_step, y_step, borders, direction, sigmas, tracks_ep, start, segment, time_tick, vdrift):

        x_start = ep.astensor(x_start)
        y_start = ep.astensor(y_start)
        z_start = ep.astensor(z_start)
        z_end = ep.astensor(z_end)
        z_start_int = ep.astensor(z_start_int)
        z_end_int = ep.astensor(z_end_int)
        z_poca = ep.astensor(z_poca)
        x_p = ep.astensor(x_p)
        y_p = ep.astensor(y_p)
        x_step = ep.astensor(x_step)
        y_step = ep.astensor(y_step)
        borders = ep.astensor(borders)
        direction = ep.astensor(direction)
        sigmas = ep.astensor(sigmas)
        tracks_ep = ep.astensor(tracks_ep)
        start = ep.astensor(start)
        segment = ep.astensor(segment)
        time_tick = ep.astensor(time_tick)

        z_sampling = self.t_sampling / 2.
        z_steps = ep.maximum(self.sampled_points, ((ep.abs(z_end_int
                                                    - z_start_int) / z_sampling)+1).astype(int))

        z_step = (z_end_int - z_start_int) / (z_steps - 1)

        iz = ep.arange(z_steps, 0, z_steps.max().item())
        z =  z_start_int[:, :, ep.newaxis] + iz[ep.newaxis, ep.newaxis, :] * z_step[..., ep.newaxis]

        t0 = (ep.abs(z - borders[:, 2, 0, ep.newaxis, ep.newaxis]) - 0.5) / vdrift

        # FIXME: this sampling is far from ideal, we should sample around the track
        # and not in a cube containing the track
        ix = ep.arange(iz, 0, self.sampled_points)
        x = x_start[:, :, ep.newaxis] + \
            ep.sign(direction[:, 0, ep.newaxis, ep.newaxis]) *\
            (ix[ep.newaxis, ep.newaxis, :] *
             x_step[:, :, ep.newaxis]  - 4 * sigmas[:, 0, ep.newaxis, ep.newaxis])
        x_dist = ep.abs(x_p - x)

        iy = ep.arange(iz, 0, self.sampled_points)
        y = y_start[:, :, ep.newaxis] + \
            ep.sign(direction[:, 1, ep.newaxis, ep.newaxis]) * \
            (iy[ep.newaxis, ep.newaxis, :] *
             y_step[:, :, ep.newaxis] - 4 * sigmas[:, 1, ep.newaxis, ep.newaxis])
        y_dist = ep.abs(y_p - y)

        charge = self.rho((x, y, z),
                          tracks_ep,
                          start, sigmas, segment) *\
                 ep.abs(x_step[:, :, ep.newaxis, ep.newaxis, ep.newaxis]) *\
                 ep.abs(y_step[:, :, ep.newaxis, ep.newaxis, ep.newaxis]) *\
                 ep.abs(z_step[:, :, ep.newaxis, ep.newaxis, ep.newaxis])

        # mask z_poca rows
        charge *= (z_poca != 0)[:, :, ep.newaxis, ep.newaxis, ep.newaxis]

        # mask x rows
        charge *= (x_dist < self.pixel_pitch / 2)[..., ep.newaxis, ep.newaxis]

        # mask y elements
        charge *= (y_dist < self.pixel_pitch / 2)[:, :, ep.newaxis, :, ep.newaxis]

        x_dist = ep.minimum(x_dist, self.pixel_pitch / 2)
        y_dist = ep.minimum(y_dist, self.pixel_pitch / 2)
        current_out = self.current_model(time_tick[:, ep.newaxis, :, ep.newaxis, ep.newaxis, ep.newaxis],
                                         t0[:, :, ep.newaxis, ep.newaxis, ep.newaxis, :],
                                         x_dist[:, :, ep.newaxis, :, ep.newaxis, ep.newaxis],
                                         y_dist[:, :, ep.newaxis, ep.newaxis, :, ep.newaxis])


        total_current = charge[:, :, ep.newaxis, ...] * current_out * self.e_charge
        
        return total_current.sum(axis=(3, 4, 5)).raw

    def tracks_current(self, pixels, npixels, tracks, time_max, fields):
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
        pixels = ep.astensor(pixels)
        tracks_ep = ep.astensor(tracks)
        # Need to trunc, otherwise gradients get funky
        time_max = ep.astensor(torch.trunc(time_max))
        it = diff_arange(0, time_max)

        # Pixel coordinates
        x_p, y_p = self.get_pixel_coordinates(pixels)
        x_p += self.pixel_pitch / 2
        y_p += self.pixel_pitch / 2

        start_coords = ep.stack([tracks_ep[:, fields.index("x_start")],
                                 tracks_ep[:, fields.index("y_start")],
                                 tracks_ep[:, fields.index("z_start")]], axis=1)
        end_coords = ep.stack([tracks_ep[:, fields.index("x_end")],
                               tracks_ep[:, fields.index("y_end")],
                               tracks_ep[:, fields.index("z_end")]], axis=1)
        cond = tracks_ep[:, fields.index("z_start")] < tracks_ep[:, fields.index("z_end")]
        start = ep.where(cond[...,ep.newaxis], start_coords, end_coords)
        end = ep.where(cond[...,ep.newaxis], end_coords, start_coords)
        segment = end - start
        length = ep.norms.l2(end, axis=1, keepdims=True)

        direction = segment / length
        sigmas = ep.stack([tracks_ep[:, fields.index("tran_diff")],
                           tracks_ep[:, fields.index("tran_diff")],
                           tracks_ep[:, fields.index("long_diff")]], axis=1)

        # The impact factor is the the size of the transverse diffusion or, if too small,
        # half the diagonal of the pixel pad
        impact_factor = ep.maximum(ep.sqrt((5 * sigmas[:, 0]) ** 2 + (5 * sigmas[:, 1]) ** 2),
                                   ep.full_like(sigmas[:, 0], sqrt(self.pixel_pitch ** 2 + self.pixel_pitch ** 2) / 2)) * 2
        z_poca, z_start, z_end = self.z_interval(start, end, x_p, y_p, impact_factor)

        z_start_int = z_start - 4 * sigmas[:, 2][...,ep.newaxis]
        z_end_int = z_end + 4 * sigmas[:, 2][...,ep.newaxis]

        x_start, y_start = self.track_point(start, direction, z_start)
        x_end, y_end = self.track_point(start, direction, z_end)

        y_step = (ep.abs(y_end - y_start) + 8 * sigmas[:, 1][...,ep.newaxis]) / (self.sampled_points - 1)
        x_step = (ep.abs(x_end - x_start) + 8 * sigmas[:, 0][...,ep.newaxis]) / (self.sampled_points - 1)

        # This was a // divide, implement?
        t_start = ep.maximum(self.time_interval[0],
                             (tracks_ep[:, fields.index("t_start")] - self.time_padding)
                             / self.t_sampling * self.t_sampling)

        time_tick = t_start[:, ep.newaxis] + it * self.t_sampling
        tpc_borders_ep = ep.from_numpy(pixels, self.tpc_borders).float32()
        borders = ep.stack([tpc_borders_ep[x.astype(int)] for x in tracks_ep[:, fields.index("pixel_plane")]])

        signals = ep.zeros(z_start, shape=(pixels.shape[0], pixels.shape[1], time_max.astype(int).item()))
        for it in range(0, z_start.shape[0], self.track_chunk):
            it_end = min(it + self.track_chunk, z_start.shape[0])
            if not self.skip_pixels:
                pix_end_range = z_start.shape[1]
            else: # ASSUMES THAT TRACK_CHUNK = 1
                pix_end_range = min(npixels[it], z_start.shape[1])
            for ip in range(0, pix_end_range, self.pixel_chunk):
                ip_end = min(ip + self.pixel_chunk, pix_end_range)
                if tracks_ep.raw.grad_fn is not None:
                    # Torch checkpointing needs torch tensors for both input and output
                    current_sum = checkpoint.checkpoint(self.calc_total_current, 
                                                     *(x_start[it:it_end, ip:ip_end].raw, y_start[it:it_end, ip:ip_end].raw, z_start.raw, 
                                                       z_end.raw, z_start_int[it:it_end, ip:ip_end].raw, z_end_int[it:it_end, ip:ip_end].raw, z_poca[it:it_end, ip:ip_end].raw, 
                                                       x_p[it:it_end, ip:ip_end].raw, y_p[it:it_end, ip:ip_end].raw, x_step[it:it_end, ip:ip_end].raw, y_step[it:it_end, ip:ip_end].raw, borders[it:it_end].raw, direction[it:it_end].raw, 
                                                       sigmas[it:it_end].raw,
                                                       tracks_ep[it:it_end, fields.index("n_electrons")].raw, start[it:it_end].raw, segment[it:it_end].raw, time_tick[it:it_end].raw, self.vdrift))
                else:
                    current_sum = self.calc_total_current( 
                                                       x_start[it:it_end, ip:ip_end].raw, y_start[it:it_end, ip:ip_end].raw, z_start.raw,
                                                       z_end.raw, z_start_int[it:it_end, ip:ip_end].raw, z_end_int[it:it_end, ip:ip_end].raw, z_poca[it:it_end, ip:ip_end].raw,
                                                       x_p[it:it_end, ip:ip_end].raw, y_p[it:it_end, ip:ip_end].raw, x_step[it:it_end, ip:ip_end].raw, y_step[it:it_end, ip:ip_end].raw, borders[it:it_end].raw, direction[it:it_end].raw,
                                                       sigmas[it:it_end].raw,
                                                       tracks_ep[it:it_end, fields.index("n_electrons")].raw, start[it:it_end].raw, segment[it:it_end].raw, time_tick[it:it_end].raw, self.vdrift)
 
                signals = ep.index_update(signals, ep.index[it:it_end, ip:ip_end, :], ep.astensor(current_sum))
        
        return signals.raw


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

        signals = ep.astensor(signals)
        track_starts = ep.astensor(track_starts)
        index_map = ep.astensor(index_map)

        # Set up index map to match with signal shape
        index = index_map[..., ep.newaxis]

        # Set up time map to match with signal shape. To implement: ep.round
        itime = ((track_starts / self.t_sampling + 0.5)[:, ep.newaxis, ep.newaxis] +
                 ep.arange(signals, 0, signals.shape[2])[ep.newaxis, ep.newaxis, :])
 
        # Each signal index now has a corresponding pixel/time index
        exp_index = ep.tile(index, (1,1,signals.shape[2]))
        exp_itime = ep.tile(itime, (1, signals.shape[1], 1))

        # Put pixel/time/signal together and flatten
        idxs = ep.stack((exp_index, exp_itime, signals), axis=-1)
        flat_idxs = idxs.reshape((-1, 3))

        # Get unique indices (return_inverse doesn't exist for ep)
        unique_idxs, idx_inv = flat_idxs[:, :2].astype(int).raw.unique(dim=0, return_inverse=True)

        unique_idxs = ep.astensor(unique_idxs)
        idx_inv = ep.astensor(idx_inv)
        # Sum over values for unique indices - scatter_add_ doesn't exist in ep. Can loop, but slow, e.g.
        #out = ep.zeros(signals, shape=(len(unique_idxs)))
        #for i in range(flat_idxs.shape[0]):
         #   out = out.index_update(idx_inv[i], out[idx_inv[i]]+flat_idxs[i, 2])
        res = ep.astensor(ep.zeros(signals, shape=(len(unique_idxs))).raw.scatter_add_(0, idx_inv.raw, flat_idxs[:, 2].raw))

        output = ep.index_update(ep.astensor(pixels_signals), (unique_idxs[:,0].astype(int),
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
