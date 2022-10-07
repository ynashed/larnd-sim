#!/usr/bin/env python3

import argparse
import sys, os
import traceback
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle

from .fit_params import ParamFitter
from .dataio import TracksDataset

def main(config):
    dataset = TracksDataset(filename=config.input_file, ntrack=config.data_sz, seed=config.data_seed, random_ntrack=config.random_ntrack, track_zlen_sel=config.track_zlen_sel)
    tracks_dataloader = DataLoader(dataset,
                                  shuffle=config.data_shuffle, 
                                  batch_size=config.batch_sz,
                                  pin_memory=True, num_workers=config.num_workers)
    # For readout noise: no_noise overrides if explicitly set to True. Otherwise, turn on noise
    # individually for target and guess
    param_fit = ParamFitter(config.param_list, dataset.get_track_fields(),
                            track_chunk=config.track_chunk, pixel_chunk=config.pixel_chunk,
                            detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                            load_checkpoint=config.load_checkpoint, lr=config.lr,
                            readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                            readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess))

    param_fit.make_target_sim(seed=config.seed)
    landscape, fname = param_fit.loss_scan_batch(tracks_dataloader, param_range=config.param_range, n_steps=config.n_steps, shuffle=config.data_shuffle, save_freq=config.save_freq)

    if config.plot:
        plt.plot(landscape['param_vals'], landscape['losses'])
        y_range = max(landscape['losses']) - min(landscape['losses'])
        x_range = max(landscape['param_vals']) - min(landscape['param_vals'])
        lr=0.02*x_range/(max(landscape['grads']))
        for i in range(len(landscape['param_vals'])):
            plt.arrow(landscape['param_vals'][i], landscape['losses'][i], -lr*landscape['grads'][i], 0, 
                      width=0.01*y_range, head_width=0.05*y_range, 
                      head_length=0.01*x_range)
        plt.xlabel(landscape['param'])
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig(fname+".pdf")
        plt.show()

    return 0, 'Fitting successful'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="param_list", default=[], nargs="+", required=True,
                        help="List of parameters to optimize. See consts_ep.py")
    parser.add_argument("--input_file", dest="input_file",
                        default="/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5",
                        help="Input data file")
    parser.add_argument("--detector_props", dest="detector_props",
                        default="larndsim/detector_properties/module0.yaml",
                        help="Path to detector properties YAML file")
    parser.add_argument("--pixel_layouts", dest="pixel_layouts",
                        default="larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml",
                        help="Path to pixel layouts YAML file")
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", type=str, default=None,
                        help="Path to checkpoint Pickle (pkl) file")
    parser.add_argument("--track_chunk", dest="track_chunk", default=1, type=int,
                        help="Track chunk size used in simulation.")
    parser.add_argument("--pixel_chunk", dest="pixel_chunk", default=1, type=int,
                        help="Pixel chunk size used in simulation.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='The number of worker threads to use for the dataloader.')
    parser.add_argument("--lr", dest="lr", default=1e1, type=float,
                        help="Learning rate -- used for all params")
    parser.add_argument("--batch_sz", dest="batch_sz", default=2, type=int,
                        help="Batch size for fitting (tracks).")
    parser.add_argument("--epochs", dest="epochs", default=100, type=int,
                        help="Number of epochs")
    parser.add_argument("--seed", dest="seed", default=2, type=int,
                        help="Random seed for target construction")
    parser.add_argument("--data_seed", dest="data_seed", default=3, type=int,
                        help="Random seed for data picking if not using the whole set")
    parser.add_argument("--data_sz", dest="data_sz", default=5, type=int,
                        help="data size for fitting (number of tracks)")
    parser.add_argument("--param_range", dest="param_range", default=None, nargs="+", type=float,
                        help="Param range for loss landscape")
    parser.add_argument("--n_steps", dest="n_steps", default=10, type=int,
                        help="Number of steps for loss landscape")
    parser.add_argument("--plot", dest="plot", default=False, action="store_true",
                        help="Makes landscape plot with arrows pointing in -grad direction")
    parser.add_argument("--no-noise", dest="no_noise", default=False, action="store_true",
                        help="Flag to turn off readout noise (both target and guess)")
    parser.add_argument("--no-noise-target", dest="no_noise_target", default=False, action="store_true",
                        help="Flag to turn off readout noise (just target, guess has noise)")
    parser.add_argument("--no-noise-guess", dest="no_noise_guess", default=False, action="store_true",
                        help="Flag to turn off readout noise (just guess, target has noise)")
    parser.add_argument("--data_shuffle", dest="data_shuffle", default=False, action="store_true",
                        help="Flag of data shuffling")
    parser.add_argument("--save_freq", dest="save_freq", default=5, type=int,
                        help="Save frequency of the result")
    parser.add_argument("--random_ntrack", dest="random_ntrack", default=False, action="store_true",
                        help="Flag of whether sampling the tracks randomly or sequentially")
    parser.add_argument("--track_zlen_sel", dest="track_zlen_sel", default=30., type=float,
                        help="Track selection requirement on the z expansion (drift axis)")

                        
    try:
        args = parser.parse_args()
        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    print(status_message)
    exit(retval)
