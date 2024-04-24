"""
Module to implement the quenching of the ionized electrons
through the detector
"""

import jax.numpy as jnp
from jax import jit
from functools import partial

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("QUENCHING MODULE PARAMETERS")

def box_model(dEdx, eField, lArDensity, alpha, beta):
    # Baller, 2013 JINST 8 P08005
    csi = beta * dEdx / (eField * lArDensity)
    return jnp.maximum(0, jnp.log(alpha + csi) / csi)

def birks_model(dEdx, eField, lArDensity, Ab, kb):
    # Amoruso, et al NIM A 523 (2004) 275
    return Ab / (1 + kb * dEdx / (eField * lArDensity))

def get_nelectrons(dE, recomb, MeVToElectrons):
    return recomb * dE * MeVToElectrons

@partial(jit, static_argnames=['fields', 'mode'])
def quench(params, tracks, mode, fields):
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

    if mode == params.box:
        recomb = box_model(tracks[:, fields.index("dEdx")], params.eField, params.lArDensity, params.alpha, params.beta)
    elif mode == params.birks:
        recomb = birks_model(tracks[:, fields.index("dEdx")], params.eField, params.lArDensity, params.Ab, params.kb)
    else:
        raise ValueError("Invalid recombination mode: must be 'box' or 'birks'")

    #TODO: n_electrons should be int, but truncation makes gradients vanish
    updated_tracks = tracks.at[:, fields.index("n_electrons")].set(get_nelectrons(tracks[:, fields.index("dE")], recomb, params.MeVToElectrons))
    return updated_tracks