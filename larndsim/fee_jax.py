"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import jax.numpy as jnp
from jax.profiler import annotate_function
from jax import grad, jit, vmap, lax, make_jaxpr, random, debug
from jax.nn import sigmoid
from functools import partial

@annotate_function
@jit
def digitize(params, integral_list):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list (:obj:`numpy.ndarray`): list of charge collected by each pixel

    Returns:
        numpy.ndarray: list of ADC values for each pixel
    """
    adcs = jnp.minimum((jnp.maximum((integral_list*params.GAIN/params.e_charge+params.V_PEDESTAL - params.V_CM), 0) \
                        * params.ADC_COUNTS/(params.V_REF-params.V_CM)+0.5), params.ADC_COUNTS)

    return adcs

@annotate_function
@jit
def get_adc_values(params, pixels_signals):
    """
    Implementation of self-trigger logic

    Args:
        pixels_signals (:obj:`numpy.ndarray`): list of induced currents for
            each pixel
        time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel
        adc_list (:obj:`numpy.ndarray`): list of integrated charges for each
            pixel
        adc_ticks_list (:obj:`numpy.ndarray`): list of the time ticks that
            correspond to each integrated charge.
    """

    #Baseline level of noise on integrated charge
    #TODO: Deal better with the rng
    key = random.PRNGKey(42)
    q_sum_base = random.normal(key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE * params.e_charge

    # Charge
    q = pixels_signals*params.t_sampling

    # Collect cumulative charge over all time ticks + add baseline noise
    q_cumsum = q.cumsum(axis=1)
    q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum

    def find_hit(carry, it):
        key, q_sum, q_cumsum = carry
        # Index of first threshold passing. For nice time axis differentiability: first find index window around threshold.
        selec_func = lambda x: jnp.where((x[1:] >= params.DISCRIMINATION_THRESHOLD) & 
                    (x[:-1] <= params.DISCRIMINATION_THRESHOLD), size=1, fill_value=q_sum.shape[1]-1)
        idx_t, = vmap(selec_func, 0, 0)(q_sum)
        idx_t = idx_t.ravel()
        idx_pix = jnp.arange(0, q_sum.shape[0])
        # Then linearly interpolate for the intersection point.
        m = (q_sum[idx_pix, idx_t]-q_sum[idx_pix, (idx_t-1)])
        b = q_sum[idx_pix, idx_t]-m*idx_t
        eps = 1e-3
        idx_val = jnp.where(m < eps*params.DISCRIMINATION_THRESHOLD, 0, (params.DISCRIMINATION_THRESHOLD - b)/(m + eps*params.DISCRIMINATION_THRESHOLD))

        ic = jnp.zeros((q_sum.shape[0],))
        ic = ic.at[idx_pix].set(idx_val)

        # End point of integration
        interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
        integrate_end = ic+interval

        #Protect against ic+integrate_end past last index
        integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)

        end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end.astype(int)]))

        # Cumulative => value at end is desired value
        q_vals = q_sum[end2d_idx] 
        q_vals_no_noise = q_cumsum[end2d_idx]

        # Uncorrelated noise
        key, = random.split(key, 1)
        extra_noise = random.normal(key, (pixels_signals.shape[0],))  * params.UNCORRELATED_NOISE_CHARGE * params.e_charge

        # Only include noise if nonzero
        adc = jnp.where(q_vals_no_noise != 0, q_vals + extra_noise, q_vals_no_noise)

        cond_adc = adc < params.DISCRIMINATION_THRESHOLD

        # Only include if passes threshold     
        adc = jnp.where(cond_adc, 0, adc)

        # Setup for next loop: baseline noise set to based on adc passing disc. threshold
        key, = random.split(key, 1)
        q_adc_pass = random.normal(key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE * params.e_charge
        key, = random.split(key, 1)
        q_adc_fail = random.normal(key, (pixels_signals.shape[0],)) * params.UNCORRELATED_NOISE_CHARGE * params.e_charge
        q_sum_base = jnp.where(cond_adc, q_adc_fail, q_adc_pass)

        # Remove charge already counted
        q_cumsum = q_cumsum - q_vals_no_noise[:, jnp.newaxis]
        q_cumsum = jnp.where(q_cumsum < 0, 0, q_cumsum)
        q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum
        

        return (key, q_sum, q_cumsum), (adc, ic)

        # Get ticks
        # adc_ticks_list = (time_ticks.max()-time_ticks.min())/time_ticks.shape[0]*ic + time_padding

        # full_adc.append(adc)
        # full_adc_ticks_list.append(adc_ticks_list)

    # List to contain adc values/ticks
    # full_adc = []
    # full_adc_ticks_list = []

    init_loop = (random.split(key, 1)[0], q_sum, q_cumsum)
    
    _, (full_adc, full_ticks) = lax.scan(find_hit, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))

    #Single iteration to detect NaNs
    # _, (full_adc, full_ticks) = find_hit(init_loop, 0)
    # full_adc = jnp.repeat(full_adc[:, jnp.newaxis], params.MAX_ADC_VALUES, axis=1)
    # full_ticks = jnp.repeat(full_ticks[:, jnp.newaxis], params.MAX_ADC_VALUES, axis=1)
    
    # full_adc_ticks_list = jnp.stack(full_adc_ticks_list, axis=1)

    return full_adc.T, full_ticks.T
    # return full_adc, full_ticks


# class fee(consts):
#     def __init__(self, readout_noise):
#         consts.__init__(self)

#         #: Maximum number of ADC values stored per pixel
#         self.MAX_ADC_VALUES = 10
#         #: Discrimination threshold
#         self.DISCRIMINATION_THRESHOLD = 7e3*self.e_charge
#         #: ADC hold delay in clock cycles
#         self.ADC_HOLD_DELAY = 15
#         #: Clock cycle time in :math:`\mu s`
#         self.CLOCK_CYCLE = 0.1
#         #: Front-end gain in :math:`mV/ke-`
#         self.GAIN = 4/1e3
#         #: Common-mode voltage in :math:`mV`
#         self.V_CM = 288
#         #: Reference voltage in :math:`mV`
#         self.V_REF = 1300
#         #: Pedestal voltage in :math:`mV`
#         self.V_PEDESTAL = 580
#         #: Number of ADC counts
#         self.ADC_COUNTS = 2**8
#         if readout_noise:
#             #: Reset noise in e-
#             self.RESET_NOISE_CHARGE = 900
#             #: Uncorrelated noise in e-
#             self.UNCORRELATED_NOISE_CHARGE = 500
#         else:
#             self.RESET_NOISE_CHARGE = 0
#             self.UNCORRELATED_NOISE_CHARGE = 0

#     def digitize(self, integral_list):
#         """
#         The function takes as input the integrated charge and returns the digitized
#         ADC counts.

#         Args:
#             integral_list (:obj:`numpy.ndarray`): list of charge collected by each pixel

#         Returns:
#             numpy.ndarray: list of ADC values for each pixel
#         """
#         integral_list = ep.astensor(integral_list)
#         adcs = ep.minimum((ep.maximum((integral_list*self.GAIN/self.e_charge+self.V_PEDESTAL - self.V_CM), 0) \
#                           * self.ADC_COUNTS/(self.V_REF-self.V_CM)+0.5), self.ADC_COUNTS)

#         return adcs.raw


#     def get_adc_values(self, pixels_signals, time_ticks, time_padding):
#         """
#         Implementation of self-trigger logic

#         Args:
#             pixels_signals (:obj:`numpy.ndarray`): list of induced currents for
#                 each pixel
#             time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel
#             adc_list (:obj:`numpy.ndarray`): list of integrated charges for each
#                 pixel
#             adc_ticks_list (:obj:`numpy.ndarray`): list of the time ticks that
#                 correspond to each integrated charge.
#         """
        
#         pixels_signals = ep.astensor(pixels_signals)
#         time_ticks = ep.astensor(time_ticks)

#         # List to contain adc values/ticks
#         full_adc = []
#         full_adc_ticks_list = []

#         #Baseline level of noise on integrated charge
#         q_sum_base = ep.normal(pixels_signals, pixels_signals.shape[0]) * self.RESET_NOISE_CHARGE * self.e_charge

#         # Charge
#         q = pixels_signals*self.t_sampling

#         # Collect cumulative charge over all time ticks + add baseline noise
#         q_cumsum = q.cumsum(axis=1)
#         q_sum = q_sum_base[:, ep.newaxis] + q_cumsum
        
#         # Main loop
#         for val in range(self.MAX_ADC_VALUES):

#             # Uncorrelated noise
#             q_noise = ep.normal(pixels_signals, pixels_signals.shape) * self.UNCORRELATED_NOISE_CHARGE * self.e_charge

#             # Find which pixel/time passes threshold
#             cond = (q_sum+q_noise >= self.DISCRIMINATION_THRESHOLD)

#             # Index of first threshold passing. For nice time axis differentiability: first find index window around threshold.
#             idx_pix, idx_t = torch.where((q_sum.raw[:, 1:] >= self.DISCRIMINATION_THRESHOLD) & 
#                         (q_sum.raw[:, :-1] <= self.DISCRIMINATION_THRESHOLD))
#             idx_pix = ep.astensor(idx_pix)
#             idx_t = ep.astensor(idx_t)
  
#             # Then linearly interpolate for the intersection point.
#             m = (q_sum[idx_pix, idx_t]-q_sum[idx_pix, (idx_t-1)])
#             b = q_sum[idx_pix, idx_t]-m*idx_t
#             idx_val = (self.DISCRIMINATION_THRESHOLD - b)/m

#             ic = ep.zeros(idx_val, q_sum.shape[0])
#             ic = ep.index_update(ic, idx_pix, idx_val)
            
#             # End point of integration
#             interval = round((3 * self.CLOCK_CYCLE + self.ADC_HOLD_DELAY * self.CLOCK_CYCLE) / self.t_sampling)
#             integrate_end = ic+interval
            
#             #Protect against ic+integrate_end past last index
#             integrate_end = ep.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)
 
#             end2d_idx = tuple(ep.stack([ep.arange(ic, 0, ic.shape[0]).astype(int), integrate_end.astype(int)]))

#             # Cumulative => value at end is desired value
#             q_vals = q_sum[end2d_idx] 
#             q_vals_no_noise = q_cumsum[end2d_idx]

#             extra_noise = ep.normal(pixels_signals, pixels_signals.shape[0])  * self.UNCORRELATED_NOISE_CHARGE * self.e_charge

#             # Only include noise if nonzero
#             adc = ep.where(q_vals_no_noise != 0, q_vals + extra_noise, q_vals_no_noise)

#             cond_adc = adc < self.DISCRIMINATION_THRESHOLD

#             # Only include if passes threshold     
#             adc = ep.where(cond_adc, 0, adc)
            
#             # Setup for next loop: baseline noise set to based on adc passing disc. threshold
#             q_adc_pass = ep.normal(pixels_signals, pixels_signals.shape[0]) * self.RESET_NOISE_CHARGE * self.e_charge
#             q_adc_fail = ep.normal(pixels_signals, pixels_signals.shape[0]) * self.UNCORRELATED_NOISE_CHARGE * self.e_charge
#             q_sum_base = ep.where(cond_adc, q_adc_fail, q_adc_pass)

#             # Remove charge already counted
#             q_cumsum = q_cumsum - q_vals_no_noise[:, ep.newaxis]
#             q_cumsum = ep.where(q_cumsum < 0, 0, q_cumsum)
#             q_sum = q_sum_base[:, ep.newaxis] + q_cumsum
           
#             # Get ticks
#             adc_ticks_list = (time_ticks.max()-time_ticks.min())/time_ticks.shape[0]*ic + time_padding

#             full_adc.append(adc)
#             full_adc_ticks_list.append(adc_ticks_list)

#         full_adc = ep.stack(full_adc, axis=1)
#         full_adc_ticks_list = ep.stack(full_adc_ticks_list, axis=1)

#         return full_adc.raw, full_adc_ticks_list.raw
