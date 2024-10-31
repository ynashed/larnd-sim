"""
Module containing constants needed by the simulation
"""

import numpy as np
import yaml
from flax import struct
import jax
import jax.numpy as jnp
import dataclasses
from types import MappingProxyType

@dataclasses.dataclass
class Params_template:
    eField: float = struct.field(pytree_node=False)
    Ab: float = struct.field(pytree_node=False)
    kb: float = struct.field(pytree_node=False)
    lifetime: float = struct.field(pytree_node=False)
    vdrift: float = struct.field(pytree_node=False)
    long_diff: float = struct.field(pytree_node=False)
    tran_diff: float = struct.field(pytree_node=False)
    tpc_borders: jax.Array = struct.field(pytree_node=False)
    box: int = struct.field(pytree_node=False)
    birks: int = struct.field(pytree_node=False)
    lArDensity: float = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)
    beta: float = struct.field(pytree_node=False)
    MeVToElectrons: float = struct.field(pytree_node=False)
    pixel_pitch: float = struct.field(pytree_node=False)
    # n_pixels: tuple = struct.field(pytree_node=False)
    n_pixels_x: tuple = struct.field(pytree_node=False)
    n_pixels_y: tuple = struct.field(pytree_node=False)
    max_radius: int = struct.field(pytree_node=False)
    max_active_pixels: int = struct.field(pytree_node=False)
    drift_length: float = struct.field(pytree_node=False)
    t_sampling: float = struct.field(pytree_node=False)
    time_interval: float = struct.field(pytree_node=False)
    time_padding: float = struct.field(pytree_node=False)
    min_step_size: float = struct.field(pytree_node=False)
    time_max: float = struct.field(pytree_node=False)
    time_window: float = struct.field(pytree_node=False)
    e_charge: float = struct.field(pytree_node=False)
    temperature: float = struct.field(pytree_node=False)
    response_bin_size: float = struct.field(pytree_node=False)
    number_pix_neighbors: int = struct.field(pytree_node=False)
    electron_sampling_resolution: float = struct.field(pytree_node=False)
    signal_length: float = struct.field(pytree_node=False)
    #: Maximum number of ADC values stored per pixel
    MAX_ADC_VALUES: int = struct.field(pytree_node=False)
    #: Discrimination threshold
    DISCRIMINATION_THRESHOLD: float = struct.field(pytree_node=False)
    #: ADC hold delay in clock cycles
    ADC_HOLD_DELAY: int = struct.field(pytree_node=False)
    #: Clock cycle time in :math:`\mu s`
    CLOCK_CYCLE: float = struct.field(pytree_node=False)
    #: Front-end gain in :math:`mV/ke-`
    GAIN: float = struct.field(pytree_node=False)
    #: Common-mode voltage in :math:`mV`
    V_CM: float = struct.field(pytree_node=False)
    #: Reference voltage in :math:`mV`
    V_REF: float = struct.field(pytree_node=False)
    #: Pedestal voltage in :math:`mV`
    V_PEDESTAL: float = struct.field(pytree_node=False)
    #: Number of ADC counts
    ADC_COUNTS: int = struct.field(pytree_node=False)
    # if readout_noise:
        #: Reset noise in e-
        # self.RESET_NOISE_CHARGE = 900
        # #: Uncorrelated noise in e-
        # self.UNCORRELATED_NOISE_CHARGE = 500
    # else:
    RESET_NOISE_CHARGE: float = struct.field(pytree_node=False)
    UNCORRELATED_NOISE_CHARGE: float = struct.field(pytree_node=False)

def build_params_class(params_with_grad):
    template_fields = dataclasses.fields(Params_template)
    # Removing the pytree_node=False for the variables requiring gradient calculation
    for param in params_with_grad:
        for field in template_fields:
            if field.name == param:
                field.metadata = MappingProxyType({})
                break
    #Dynamically creating the class from the fields and passing it to struct that will itself pass it to dataclass, ouf...
    base_class = type("Params", (object, ), {field.name: field for field in template_fields})
    base_class.__annotations__ = {field.name: field.type for field in template_fields}
    return struct.dataclass(base_class)

def load_detector_properties(params_cls, detprop_file, pixel_file):
        """
        The function loads the detector properties and
        the pixel geometry YAML files and stores the constants
        as global variables

        Args:
            detprop_file (str): detector properties YAML
                filename
            pixel_file (str): pixel layout YAML filename
        """

        params_dict = {
            "eField": 0.50,
            "Ab": 0.8,
            "kb": 0.0486,
            "vdrift": 0.1648,
            "lifetime": 2.2e3,
            "long_diff": 4.0e-6,
            "tran_diff": 8.8e-6,
            "box": 1,
            "birks": 2,
            "lArDensity": 1.38,
            "alpha": 0.93,
            "beta": 0.207,
            "MeVToElectrons": 4.237e+04,
            "temperature": 87.17,
            "max_active_pixels": 0,
            "max_radius": 0,
            "min_step_size": 0.001, #cm
            "time_max": 0,
            "time_window": 189.1, #us,
            "e_charge": 1.602e-19,
            "t_sampling": 0.1,
            "time_padding": 190,
            "response_bin_size": 0.04434,
            "number_pix_neighbors": 1,
            "electron_sampling_resolution": 0.001,
            "signal_length": 150,
            "MAX_ADC_VALUES": 10,
            "DISCRIMINATION_THRESHOLD": 7e3*1.602e-19,
            "ADC_HOLD_DELAY": 15,
            "CLOCK_CYCLE": 0.1,
            "GAIN": 4e-3,
            "V_CM": 288,
            "V_REF": 1300,
            "V_PEDESTAL": 580,
            "ADC_COUNTS": 2**8,
            "RESET_NOISE_CHARGE": 0,
            "UNCORRELATED_NOISE_CHARGE": 0,
        }

        mm2cm = 0.1
        params_dict['tpc_borders'] = np.zeros((0, 3, 2))
        params_dict['tile_borders'] = np.zeros((2,2))

        with open(detprop_file) as df:
            detprop = yaml.load(df, Loader=yaml.FullLoader)

        params_dict['tpc_centers'] = np.array(detprop['tpc_centers'])
        params_dict['tpc_centers'][:, [2, 0]] = params_dict['tpc_centers'][:, [0, 2]]

        params_dict['time_interval'] = np.array(detprop['time_interval'])

        params_dict['drift_length'] = detprop['drift_length']
        params_dict['vdrift_static'] = detprop['vdrift_static']

        params_dict['eField'] = detprop['eField']
        params_dict['vdrift'] = detprop['vdrift']
        params_dict['lifetime'] = detprop['lifetime']
        params_dict['MeVToElectrons'] = detprop['MeVToElectrons']
        params_dict['Ab'] = detprop['Ab']
        params_dict['kb'] = detprop['kb']
        params_dict['long_diff'] = detprop['long_diff']
        params_dict['tran_diff'] = detprop['tran_diff']
        if 'temperatrue' in detprop:
            params_dict['temperature'] = detprop['temperature']

        with open(pixel_file, 'r') as pf:
            tile_layout = yaml.load(pf, Loader=yaml.FullLoader)

        params_dict['pixel_pitch'] = tile_layout['pixel_pitch'] * mm2cm
        chip_channel_to_position = tile_layout['chip_channel_to_position']
        params_dict['pixel_connection_dict'] = {tuple(pix): (chip_channel//1000,chip_channel%1000) for chip_channel, pix in chip_channel_to_position.items()}
        params_dict['tile_chip_to_io'] = tile_layout['tile_chip_to_io']

        params_dict['xs'] = np.array(list(chip_channel_to_position.values()))[:,0] * params_dict['pixel_pitch']
        params_dict['ys'] = np.array(list(chip_channel_to_position.values()))[:,1] * params_dict['pixel_pitch']
        params_dict['tile_borders'][0] = [-(max(params_dict['xs'])+params_dict['pixel_pitch'])/2, (max(params_dict['xs'])+params_dict['pixel_pitch'])/2]
        params_dict['tile_borders'][1] = [-(max(params_dict['ys'])+params_dict['pixel_pitch'])/2, (max(params_dict['ys'])+params_dict['pixel_pitch'])/2]

        params_dict['tile_positions'] = np.array(list(tile_layout['tile_positions'].values())) * mm2cm
        params_dict['tile_orientations'] = np.array(list(tile_layout['tile_orientations'].values()))
        tpcs = np.unique(params_dict['tile_positions'][:,0])
        params_dict['tpc_borders'] = np.zeros((len(tpcs), 3, 2))

        for itpc,tpc_id in enumerate(tpcs):
            this_tpc_tile = params_dict['tile_positions'][params_dict['tile_positions'][:,0] == tpc_id]
            this_orientation = params_dict['tile_orientations'][params_dict['tile_positions'][:,0] == tpc_id]
            x_border = min(this_tpc_tile[:,2])+params_dict['tile_borders'][0][0]+params_dict['tpc_centers'][itpc][0], \
                       max(this_tpc_tile[:,2])+params_dict['tile_borders'][0][1]+params_dict['tpc_centers'][itpc][0]
            y_border = min(this_tpc_tile[:,1])+params_dict['tile_borders'][1][0]+params_dict['tpc_centers'][itpc][1], \
                       max(this_tpc_tile[:,1])+params_dict['tile_borders'][1][1]+params_dict['tpc_centers'][itpc][1]
            z_border = min(this_tpc_tile[:,0])+params_dict['tpc_centers'][itpc][2], \
                       max(this_tpc_tile[:,0])+detprop['drift_length']*this_orientation[:,0][0]+params_dict['tpc_centers'][itpc][2]

            params_dict['tpc_borders'][itpc] = (x_border, y_border, z_border)
     
        #: Number of pixels per axis
        # params_dict['n_pixels'] = len(np.unique(params_dict['xs']))*2, len(np.unique(params_dict['ys']))*4
        params_dict['n_pixels_x'] = len(np.unique(params_dict['xs']))*2
        params_dict['n_pixels_y'] = len(np.unique(params_dict['ys']))*4

        params_dict['n_pixels_per_tile'] = len(np.unique(params_dict['xs'])), len(np.unique(params_dict['ys']))

        params_dict['tile_map'] = ((7,5,3,1),(8,6,4,2)),((16,14,12,10),(15,13,11,9))
        params_dict['tpc_borders'] = jnp.asarray(params_dict['tpc_borders'])
        filtered_dict = {key: value for key, value in params_dict.items() if key in params_cls.__match_args__}
        return params_cls(**filtered_dict)