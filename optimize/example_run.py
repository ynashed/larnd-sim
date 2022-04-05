#!/usr/bin/env python3

import argparse
import os
import sys
import traceback
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from fit_params import ParamFitter
from dataio import TracksDataset

def main(config):
    dataset = TracksDataset(filename=config.input_file, ntrack=config.data_sz)
    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    tracks_dataloader = DataLoader(dataset,
                                   shuffle=(sampler is None),
                                   sampler=sampler, batch_size=config.batch_sz,
                                   pin_memory=True, num_workers=config.num_workers)
    param_fit = ParamFitter(config.param_list, dataset.get_track_fields(),
                            track_chunk=config.track_chunk, pixel_chunk=config.pixel_chunk, job_id=config.job_id,
                            local_rank=config.local_rank, world_rank=config.world_rank, world_size=config.world_size,
                            detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                            load_checkpoint=config.load_checkpoint, lr=config.lr)
    param_fit.make_target_sim(seed=config.seed)
    param_fit.fit(tracks_dataloader, sampler, epochs=config.epochs)

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
    parser.add_argument("--data_sz", dest="data_sz", default=5, type=int,
                        help="data size for fitting (number of tracks)")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--job_id', type=str, default='000')

    try:
        args = parser.parse_args()
        args_dict = vars(args)
        args_dict['world_size'] = 1
        if 'WORLD_SIZE' in os.environ:
            args_dict['world_size'] = int(os.environ['WORLD_SIZE'])

        args_dict['local_rank'] = args.local_rank
        args_dict['world_rank'] = 0
        if args_dict['world_size'] > 1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            args_dict['world_rank'] = dist.get_rank()
            args_dict['global_batch_size'] = args.batch_sz
            args_dict['batch_sz'] = int(args.batch_sz // args.world_size)
        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    print(status_message)
    exit(retval)
