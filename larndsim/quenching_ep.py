"""
Module to implement the quenching of the ionized electrons
through the detector
"""

import eagerpy as ep

from . import consts

import logging

logging.basicConfig()
logger = logging.getLogger('quenching')
logger.setLevel(logging.WARNING)
logger.info("QUENCHING MODULE PARAMETERS")


def quench(tracks, mode, fields):
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

    if mode == consts.box:
        # Baller, 2013 JINST 8 P08005
        csi = consts.beta * dEdx / (consts.eField * consts.lArDensity)
        recomb = ep.maximum(0, ep.log(consts.alpha + csi) / csi)
    elif mode == consts.birks:
        # Amoruso, et al NIM A 523 (2004) 275
        recomb = consts.Ab / (1 + consts.kb * dEdx / (consts.eField * consts.lArDensity))
    else:
        raise ValueError("Invalid recombination mode: must be 'box' or 'birks'")

    if ep.isnan(recomb).any():
        raise RuntimeError("Invalid recombination value")

    tracks[:, fields.index("n_electrons")] = (recomb * dE * consts.MeVToElectrons).raw

