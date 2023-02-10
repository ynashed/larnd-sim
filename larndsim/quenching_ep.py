"""
Module to implement the quenching of the ionized electrons
through the detector
"""

import eagerpy as ep

from .consts_ep import consts
from profiling.profiling import to_profile

import logging
from profiling.profiling import memprof

logging.basicConfig()
logger = logging.getLogger('quenching')
logger.setLevel(logging.WARNING)
logger.info("QUENCHING MODULE PARAMETERS")

class quench(consts):
    def __init__(self):
        consts.__init__(self)

    @to_profile
    def quench(self, tracks, mode, fields):
        """
        This function takes as input an (unstructured) array of track segments and calculates
        the number of electrons that reach the anode plane after recombination.
        It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
        Birks (Amoruso, et al NIM A 523 (2004) 275).

        Args:
            tracks (:obj:`numpy.ndarray`, `pyTorch/Tensorflow/JAX Tensor`): array containing the tracks segment information
            mode (int): recombination model.
            fields (list): an ordered string list of field/column name of the tracks structured array
        """
        tracks_ep = ep.astensor(tracks)
        dEdx = tracks_ep[:, fields.index("dEdx")]
        dE = tracks_ep[:, fields.index("dE")]

        if mode == self.box:
            # Baller, 2013 JINST 8 P08005
            csi = self.beta * dEdx / (self.eField * self.lArDensity)
            recomb = ep.maximum(0, ep.log(self.alpha + csi) / csi)
        elif mode == self.birks:
            # Amoruso, et al NIM A 523 (2004) 275
            recomb = self.Ab / (1 + self.kb * dEdx / (self.eField * self.lArDensity))
        else:
            raise ValueError("Invalid recombination mode: must be 'box' or 'birks'")

        if ep.isnan(recomb).any():
            raise RuntimeError("Invalid recombination value")
        
	#TODO: n_electrons should be int, but truncation makes gradients vanish
        tracks_ep = ep.index_update(tracks_ep, 
                                    ep.index[:, fields.index("n_electrons")], (recomb * dE * self.MeVToElectrons))
        return tracks_ep.raw

