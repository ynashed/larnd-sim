#!/usr/bin/env python3

import argparse
import yaml
import sys, os
import traceback
from torch.utils.data import DataLoader

from .fit_params import ParamFitter
from .dataio import TracksDataset

def make_param_list(config):
    if len(config.param_list) == 1 and os.path.splitext(config.param_list[0])[1] == ".yaml":
        with open(config.param_list[0], 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
        for key in config_dict.keys():
            print(f"Setting lr {config_dict[key]} for {key}")
        param_list = config_dict
    else:
        param_list = config.param_list
    return param_list


def main(config):
    dataset = TracksDataset(filename=config.input_file, ntrack=config.data_sz, seed=config.data_seed, random_ntrack=config.random_ntrack, 
                            track_zlen_sel=config.track_zlen_sel, track_z_bound=config.track_z_bound, max_batch_len=config.max_batch_len)

    batch_sz = config.batch_sz
    if config.max_batch_len is not None and batch_sz != 1:
        print("Need batch size == 1 for splitting in dx chunks. Setting now...")
        batch_sz = 1

    tracks_dataloader = DataLoader(dataset,
                                  shuffle=config.data_shuffle, 
                                  batch_size=batch_sz,
                                  pin_memory=True, num_workers=config.num_workers)

    # For readout noise: no_noise overrides if explicitly set to True. Otherwise, turn on noise
    # individually for target and guess
    param_list = make_param_list(config)
    param_fit = ParamFitter(param_list, dataset.get_track_fields(),
                            track_chunk=config.track_chunk, pixel_chunk=config.pixel_chunk,
                            detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                            load_checkpoint=config.load_checkpoint, lr=config.lr, 
                            readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                            readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                            out_label=config.out_label, norm_scheme=config.norm_scheme, max_clip_norm_val=config.max_clip_norm_val,
                            fit_diffs=config.fit_diffs, optimizer_fn=config.optimizer_fn,
                            no_adc=config.no_adc, loss_fn=config.loss_fn)
    param_fit.make_target_sim(seed=config.seed, fixed_range=config.fixed_range)
    param_fit.fit(tracks_dataloader, epochs=config.epochs, iterations=config.iterations, shuffle=config.data_shuffle, save_freq=config.save_freq)

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
    parser.add_argument("--track_zlen_sel", dest="track_zlen_sel", default=2., type=float,
                        help="Track selection requirement on the z expansion (drift axis)")
    parser.add_argument("--track_z_bound", dest="track_z_bound", default=28., type=float,
                        help="Set z bound to keep healthy set of tracks")
    parser.add_argument("--out_label", dest="out_label", default="",
                        help="Label for output pkl file")
    parser.add_argument("--fixed_range", dest="fixed_range", default=None, type=float,
                        help="Construct target by sampling in a certain range (fraction of nominal)")
    parser.add_argument("--norm_scheme", dest="norm_scheme", default="divide",
                        help="Normalization scheme to use for params. Right now, divide (by nom) and standard (subtract mean, div by variance)")
    parser.add_argument("--max_clip_norm_val", dest="max_clip_norm_val", default=None, type=float,
                        help="If passed, does gradient clipping (norm)")
    parser.add_argument("--fit_diffs", dest="fit_diffs", default=False, action="store_true",
                        help="Turns on fitting of differences rather than direct fitting of values")
    parser.add_argument("--optimizer_fn", dest="optimizer_fn", default="Adam",
                        help="Choose optimizer function (here Adam vs SGD")
    parser.add_argument("--no_adc", dest="no_adc", default=False, action="store_true",
                        help="Don't include ADC in loss (e.g. for vdrift)")
    parser.add_argument("--iterations", dest="iterations", default=None, type=int,
                        help="Number of iterations to run. Overrides epochs.")
    parser.add_argument("--loss_fn", dest="loss_fn", default=None,
                        help="Loss function to use. Named options are SDTW and space_match.")
    parser.add_argument("--max_batch_len", dest="max_batch_len", default=None, type=float,
                        help="Max dx [cm] per batch. If passed, will add tracks to batch until overflow, splitting where needed")
    try:
        args = parser.parse_args()
        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    print(status_message)
    exit(retval)
