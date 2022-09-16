"""
Module containing constants needed by the simulation
"""

import numpy as np
import yaml
import torch
from .ranges import ranges

class manage_diff:
    '''
    Descriptor class to allow for easy switching between full values vs nominal + diffs
    '''
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name
        
    def __get__(self, obj, objtype=None):
        if f'{self.public_name}_diff' in obj.__dict__.keys():
            nom = getattr(obj, f'{self.public_name}_nom')
            diff = getattr(obj, f'{self.public_name}_diff')
            return nom + diff
        else:
            return getattr(obj, self.private_name)
        
    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)

class consts:
    ## There might be a better way to do this, but for now give the option to fit diffs for usual param set
    eField = manage_diff()
    Ab = manage_diff()
    kb = manage_diff()
    lifetime = manage_diff()
    vdrift = manage_diff()
    long_diff = manage_diff()
    tran_diff = manage_diff()

    def __init__(self):
        ## Turn smoothing on/off to help gradients
        self.smooth = True

        ## Detector constants
        #: Liquid argon density in :math:`g/cm^3`
        self.lArDensity = 1.38 # g/cm^3
        #: Electric field magnitude in :math:`kV/cm`
        self.eField = 0.50 # kV/cm

        ## Unit Conversions
        self.MeVToElectrons = 4.237e+04

        ## Physical params
        #: Recombination :math:`\alpha` constant for the Box model
        self.alpha = 0.93
        #: Recombination :math:`\beta` value for the Box model in :math:`(kV/cm)(g/cm^2)/MeV`
        self.beta = 0.207 #0.3 (MeV/cm)^-1 * 1.383 (g/cm^3)* 0.5 (kV/cm), R. Acciarri et al JINST 8 (2013) P08005
        #: Recombination :math:`A_b` value for the Birks Model
        self.Ab = 0.800
        #: Recombination :math:`k_b` value for the Birks Model in :math:`(kV/cm)(g/cm^2)/MeV`
        self.kb = 0.0486 # g/cm2/MeV Amoruso, et al NIM A 523 (2004) 275
        #: Electron charge in Coulomb
        self.e_charge = 1.602e-19

        ## TPC params
        #: Drift velocity in :math:`cm/\mu s`
        self.vdrift = 0.1648 # cm / us,
        #: Electron lifetime in :math:`\mu s`
        self.lifetime = 2.2e3 # us,
        #: Time sampling in :math:`\mu s`
        self.t_sampling = 0.1 # us
        #: Drift time window in :math:`\mu s`
        self.time_interval = (0, 200.) # us
        #: Signal time window padding in :math:`\mu s`
        self.time_padding = 5
        #: Number of sampled points for each segment slice
        self.sampled_points = 30
        #: Longitudinal diffusion coefficient in :math:`cm^2/\mu s`
        self.long_diff = 4.0e-6 # cm * cm / us
        #: Transverse diffusion coefficient in :math:`cm^2/\mu s`
        self.tran_diff = 8.8e-6 # cm * cm / us
        #: Numpy array containing all the time ticks in the drift time window
        self.time_ticks = np.linspace(self.time_interval[0],
                                 self.time_interval[1],
                                 int(round(self.time_interval[1]-self.time_interval[0])/self.t_sampling)+1)
        ## Quenching parameters
        self.box = 1
        self.birks = 2

        self.mm2cm = 0.1
        self.cm2mm = 10

        self.drift_length = 30.27225 # cm
        self.vdrift_static = 0.1587 # cm / us

        self.tpc_borders = np.zeros((0, 3, 2))
        self.tile_borders = np.zeros((2,2))
        self.tile_size = np.zeros(3)
        self.n_pixels = 0, 0
        self.n_pixels_per_tile = 0, 0
        self.pixel_connection_dict = {}
        self.pixel_pitch = 0
        self.tile_positions = {}
        self.tile_orientations = {}
        self.tile_map = ()
        self.tile_chip_to_io = {}

        self.variable_types = {
            "eventID": "u4",
            "z_end": "f4",
            "trackID": "u4",
            "tran_diff": "f4",
            "z_start": "f4",
            "x_end": "f4",
            "y_end": "f4",
            "n_electrons": "u4",
            "pdgId": "i4",
            "x_start": "f4",
            "y_start": "f4",
            "t_start": "f4",
            "dx": "f4",
            "long_diff": "f4",
            "pixel_plane": "u4",
            "t_end": "f4",
            "dEdx": "f4",
            "dE": "f4",
            "t": "f4",
            "y": "f4",
            "x": "f4",
            "z": "f4"
        }

        self.anode_layout = (2,4)
        self.xs = 0
        self.ys = 0

    def load_detector_properties(self, detprop_file, pixel_file):
        """
        The function loads the detector properties and
        the pixel geometry YAML files and stores the constants
        as global variables

        Args:
            detprop_file (str): detector properties YAML
                filename
            pixel_file (str): pixel layout YAML filename
        """

        with open(detprop_file) as df:
            detprop = yaml.load(df, Loader=yaml.FullLoader)

        self.tpc_centers = np.array(detprop['tpc_centers'])
        self.tpc_centers[:, [2, 0]] = self.tpc_centers[:, [0, 2]]

        self.time_interval = np.array(detprop['time_interval'])

        self.drift_length = detprop['drift_length']
        self.vdrift_static = detprop['vdrift_static']

        self.eField = detprop['eField']
        self.vdrift = detprop['vdrift']
        self.lifetime = detprop['lifetime']
        self.MeVToElectrons = detprop['MeVToElectrons']
        self.Ab = detprop['Ab']
        self.kb = detprop['kb']
        self.long_diff = detprop['long_diff']
        self.tran_diff = detprop['tran_diff']

        with open(pixel_file, 'r') as pf:
            tile_layout = yaml.load(pf, Loader=yaml.FullLoader)

        self.pixel_pitch = tile_layout['pixel_pitch'] * self.mm2cm
        chip_channel_to_position = tile_layout['chip_channel_to_position']
        self.pixel_connection_dict = {tuple(pix): (chip_channel//1000,chip_channel%1000) for chip_channel, pix in chip_channel_to_position.items()}
        self.tile_chip_to_io = tile_layout['tile_chip_to_io']

        self.xs = np.array(list(chip_channel_to_position.values()))[:,0] * self.pixel_pitch
        self.ys = np.array(list(chip_channel_to_position.values()))[:,1] * self.pixel_pitch
        self.tile_borders[0] = [-(max(self.xs)+self.pixel_pitch)/2, (max(self.xs)+self.pixel_pitch)/2]
        self.tile_borders[1] = [-(max(self.ys)+self.pixel_pitch)/2, (max(self.ys)+self.pixel_pitch)/2]

        self.tile_positions = np.array(list(tile_layout['tile_positions'].values())) * self.mm2cm
        self.tile_orientations = np.array(list(tile_layout['tile_orientations'].values()))
        tpcs = np.unique(self.tile_positions[:,0])
        self.tpc_borders = np.zeros((len(tpcs), 3, 2))

        for itpc,tpc_id in enumerate(tpcs):
            this_tpc_tile = self.tile_positions[self.tile_positions[:,0] == tpc_id]
            this_orientation = self.tile_orientations[self.tile_positions[:,0] == tpc_id]
            x_border = min(this_tpc_tile[:,2])+self.tile_borders[0][0]+self.tpc_centers[itpc][0], \
                       max(this_tpc_tile[:,2])+self.tile_borders[0][1]+self.tpc_centers[itpc][0]
            y_border = min(this_tpc_tile[:,1])+self.tile_borders[1][0]+self.tpc_centers[itpc][1], \
                       max(this_tpc_tile[:,1])+self.tile_borders[1][1]+self.tpc_centers[itpc][1]
            z_border = min(this_tpc_tile[:,0])+self.tpc_centers[itpc][2], \
                       max(this_tpc_tile[:,0])+detprop['drift_length']*this_orientation[:,0][0]+self.tpc_centers[itpc][2]

            self.tpc_borders[itpc] = (x_border, y_border, z_border)
     
        #: Number of pixels per axis
        self.n_pixels = len(np.unique(self.xs))*2, len(np.unique(self.ys))*4
        self.n_pixels_per_tile = len(np.unique(self.xs)), len(np.unique(self.ys))

        self.tile_map = ((7,5,3,1),(8,6,4,2)),((16,14,12,10),(15,13,11,9))

    def track_gradients(self, param_list, fit_diffs = False):
        self.fit_diffs = fit_diffs
        for param in param_list:
            try:
                if fit_diffs:
                    nom_val = ranges[param]['nom']
                    diff_val = getattr(self, param) - getattr(self, f'{param}_nom')
                    setattr(self, f'{param}_nom', torch.tensor(nom_val))
                    setattr(self, f'{param}_diff', torch.tensor(diff_val, requires_grad=True))
                else:
                    attr = getattr(self, param)
                    setattr(self, param, torch.tensor(float(attr), requires_grad=True))
            except:
                raise ValueError(f"Unable to track gradients for param {param}")
