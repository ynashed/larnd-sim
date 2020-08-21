
"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""

import numba as nb
import numpy as np

from numba import cuda
from math import pi, ceil, sqrt, erf, exp
from .consts import *
from . import indeces as i

@cuda.jit(device=True)
def z_interval(start_point, end_point, x_p, y_p, tolerance):
    """Here we calculate the interval in the drift direction for the pixel pID
    using the impact factor"""

    if start_point[0] > end_point[0]:
        start = end_point
        end = start_point
    elif start_point[0] < end_point[0]:
        start = start_point
        end = end_point
    else: # Limit case that we should probably manage better
        return 0, 0, 0

    xs, ys = start[0], start[1]
    xe, ye = end[0], end[1]

    m = (ye - ys) / (xe - xs)
    q = (xe * ys - xs * ye) / (xe - xs)

    a, b, c = m, -1, q

    x_poca = (b*(b*x_p-a*y_p) - a*c)/(a*a+b*b)

    length = sqrt((end[0]-start[0])**2+(end[1]-start[1])**2+(end[2]-start[2])**2)
    dir3D = (end[0] - start[0])/length, (end[1] - start[1])/length, (end[2] - start[2])/length

    if x_poca < start[0]:
        doca = sqrt((x_p - start[0])**2 + (y_p - start[1])**2)
        x_poca = start[0]
    elif x_poca > end[0]:
        doca = sqrt((x_p - end[0])**2 + (y_p - end[1])**2)
        x_poca = end[0]
    else:
        doca = abs(a*x_p+b*y_p+c)/sqrt(a*a+b*b)

    y_poca = start[1] + (x_poca - start[0])/dir3D[0]*dir3D[1]
    z_poca = start[2] + (x_poca - start[0])/dir3D[0]*dir3D[2]
    plusDeltaZ, minusDeltaZ = 0, 0

    if tolerance > doca:
        length2D = sqrt((xe-xs)**2 + (ye-ys)**2)
        dir2D = (end[0]-start[0])/length2D, (end[1]-start[1])/length2D
        deltaL2D = sqrt(tolerance**2 - doca**2) # length along the track in 2D

        x_plusDeltaL = x_poca + deltaL2D*dir2D[0] # x coordinates of the tolerance range
        x_minusDeltaL = x_poca - deltaL2D*dir2D[0]
        plusDeltaL = (x_plusDeltaL - start[0])/dir3D[0] # length along the track in 3D
        minusDeltaL = (x_minusDeltaL - start[0])/dir3D[0] # of the tolerance range

        plusDeltaZ = start[2] + dir3D[2] * plusDeltaL # z coordinates of the
        minusDeltaZ = start[2] + dir3D[2] * minusDeltaL # tolerance range

        return z_poca, min(minusDeltaZ, plusDeltaZ), max(minusDeltaZ, plusDeltaZ)
    else:
        return 0, 0, 0

@cuda.jit(device=True)
def _b(x, y, z, start, sigmas, segment, Deltar):
    return -((x-start[0]) / (sigmas[0]*sigmas[0]) * (segment[0]/Deltar) + \
             (y-start[1]) / (sigmas[1]*sigmas[1]) * (segment[1]/Deltar) + \
             (z-start[2]) / (sigmas[2]*sigmas[2]) * (segment[2]/Deltar))

@cuda.jit(device=True)
def rho(x, y, z, q, start, sigmas, segment):
    Deltax, Deltay, Deltaz = segment[0], segment[1], segment[2]
    Deltar = sqrt(Deltax**2+Deltay**2+Deltaz**2)
    a = ((Deltax/Deltar) * (Deltax/Deltar) / (2*sigmas[0]*sigmas[0]) + \
         (Deltay/Deltar) * (Deltay/Deltar) / (2*sigmas[1]*sigmas[1]) + \
         (Deltaz/Deltar) * (Deltaz/Deltar) / (2*sigmas[2]*sigmas[2]))
    factor = q/Deltar/(sigmas[0]*sigmas[1]*sigmas[2]*sqrt(8*pi*pi*pi))
    sqrt_a_2 = 2*sqrt(a)

    b = _b(x, y, z, start, sigmas, segment, Deltar)

    delta = (x-start[0])*(x-start[0])/(2*sigmas[0]*sigmas[0]) + \
            (y-start[1])*(y-start[1])/(2*sigmas[1]*sigmas[1]) + \
            (z-start[2])*(z-start[2])/(2*sigmas[2]*sigmas[2])

    expo = exp(b*b/(4*a) - delta)

    integral = sqrt(pi) * \
               (-erf(b/sqrt_a_2) + erf((b + 2*a*Deltar)/sqrt_a_2)) / \
               sqrt_a_2

    return expo * integral * factor 


@cuda.jit(device=True)
def distance_attenuation(x_p, y_p, xl, yl, q, start, point, sigmas, segment, padding, slice_size):
    summed_weight = 0

    x_step = 2 * padding / (slice_size - 1)
    y_step = 2 * padding / (slice_size - 1)
    
    for ix in range(slice_size):
        for iy in range(slice_size):
            x = point[0] - padding + ix*x_step
            y = point[1] - padding + iy*y_step
            xv = xl - padding + ix*x_step
            yv = yl - padding + iy*y_step
            summed_weight += rho(x, y, point[2], q, start, sigmas, segment) \
                             * exp(-1e2*sqrt((x_p - xv)**2+(y_p - yv)**2)) \
                             * x_step * y_step

    return summed_weight

@cuda.jit(device=True)
def track_point(start, end, direction, z):
    l = (z - start[2]) / direction[2]
    l_max = (end[2] - start[2]) / direction[2]
    l = min(max(l, 0), l_max)
    xl = start[0] + l * direction[0]
    yl = start[1] + l * direction[1]
    
    return xl, yl

@cuda.jit(device=True)
def current_signal(time, t0):
    A = 1
    B = 5
    return A*exp((time-t0)/B)

            
@nb.jit
def pixel_response(pixel_signals, anode_t):
    current = np.zeros_like(anode_t)

    for signal in pixel_signals:
        current[(anode_t >= signal[0]) & (anode_t <= signal[1])] += signal[2]

    return current


float_array = nb.types.float32[::1]
pixelID_type = nb.types.Tuple((nb.int64, nb.int64))
signal_type = nb.types.ListType(nb.types.Tuple((nb.float64, nb.float64, float_array)))

@nb.njit
def join_pixel_signals(signals, pixels):
    active_pixels = nb.typed.Dict.empty(key_type=pixelID_type,
                                        value_type=signal_type)
    t_start = time_interval[0]
    t_end = time_interval[1]
    
    for itrk in range(signals.shape[0]):
        for ipix in range(signals.shape[1]):
            pID = int(pixels[itrk][ipix][0]), int(pixels[itrk][ipix][1])
            if pID[0] < 0 or pID[1] < 0:
                continue

            signal = signals[itrk][ipix]
            if not signal.any():
                continue

            if pID in active_pixels:
                active_pixels[pID].append((t_start, t_end, signal))
            else:
                this_pixel_signal = nb.typed.List()
                this_pixel_signal.append((t_start, t_end, signal))
                active_pixels[pID] = this_pixel_signal

    return active_pixels

@cuda.jit
def tracks_current(signals, pixels, pixel_size, tracks, time_interval, time_padding, t_sampling, slice_size):
    itrk,ipix,it = cuda.grid(3)
    
    if itrk < signals.shape[0] and ipix < signals.shape[1] and it < signals.shape[2]:
        t = tracks[itrk]
        pID = pixels[itrk][ipix]

        if pID[0] >= 0 and pID[1] >= 0:
            x_p = pID[0] * pixel_size[0] + tpc_borders[0][0] + pixel_size[0] / 2
            y_p = pID[1] * pixel_size[1] + tpc_borders[1][0] + pixel_size[1] / 2

            impact_factor = 1.5 * sqrt(pixel_size[0]**2 + pixel_size[1]**2)

            start = (t[i.x_start], t[i.y_start], t[i.z_start])
            end = (t[i.x_end], t[i.y_end], t[i.z_end])
            sigmas = (t[i.tran_diff], t[i.tran_diff], t[i.long_diff])
            segment = (end[0]-start[0],end[1]-start[1],end[2]-start[2])
            length = sqrt(segment[0]**2+segment[1]**2+segment[2]**2)
            direction = (segment[0]/length, segment[1]/length, segment[2]/length)

            mid_point = (t[i.x_end] + t[i.x_start])/2., (t[i.y_end] + t[i.y_start])/2., (t[i.z_end] + t[i.z_start])/2.

            z_sampling = t_sampling * vdrift
            padding = t[i.tran_diff] * 2.5
            z_poca, z_start, z_end = z_interval(start, end, x_p, y_p, impact_factor)

            if z_start != 0 and z_end != 0:
                z_range_up = ceil(abs(z_end-z_poca)/z_sampling)+1
                z_range_down = ceil(abs(z_poca-z_start)/z_sampling)+1
                z_step = (z_end-z_poca)/(z_range_up-1)

            for iz in range(z_range_up):
                z_coord = z_poca + iz*z_step
                xl, yl = track_point(start, end, direction, z_coord)
                t0 = (z_coord - tpc_borders[2][0])/vdrift
                if time_interval[it] < t0:
                    signals[itrk][ipix][it] += current_signal(time_interval[it], t0) \
                                               * distance_attenuation(x_p, y_p, xl, yl, 
                                                                      t[i.n_electrons], 
                                                                      start, mid_point, sigmas, 
                                                                      segment, padding, slice_size) * z_step

            for iz in range(1,z_range_down):
                z_coord = z_poca - iz*z_step
                xl, yl = track_point(start, end, direction, z_coord)
                t0 = (z_coord - tpc_borders[2][0])/vdrift
                if time_interval[it] < t0:
                    signals[itrk][ipix][it] += current_signal(time_interval[it], t0) \
                                               * distance_attenuation(x_p, y_p, xl, yl, 
                                                                      t[i.n_electrons], 
                                                                      start, mid_point, sigmas, 
                                                                      segment, padding, slice_size) * z_step
@nb.jit
def pixel_from_coordinates(x, y, n_pixels):
    """This function returns the ID of the pixel that covers the specified point

    Args:
        x (float): x coordinate
        y (float): y coordinate
        n_pixels (int): number of pixels for each axis

    Returns:
        tuple: the pixel ID
    """

    x_pixel = np.linspace(tpc_borders[0][0], tpc_borders[0][1], n_pixels)
    y_pixel = np.linspace(tpc_borders[1][0], tpc_borders[1][1], n_pixels)
    return np.digitize(x, x_pixel), np.digitize(y, y_pixel)
