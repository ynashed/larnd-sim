#!/usr/bin/env python3

import argparse
import yaml
import sys, os
import traceback
from torch.utils.data import DataLoader
import shutil
from time import time
from math import ceil

from .dataio import TracksDataset, structured_from_torch
from .utils import get_id_map
from larndsim import quenching, drifting, consts

class Simulation:
    def __init__(self, track_dtypes, track_chunk, pixel_chunk,
                 detector_props, pixel_layouts,
                 lr=None, optimizer=None, loss_fn=None, readout_noise_target=True, readout_noise_guess=False, 
                 out_label="", norm_scheme="divide") -> None:

        self.out_label = out_label
        self.norm_scheme = norm_scheme
        self.device = 'cuda'

        self.track_dtypes = track_dtypes

        # Simulation object for target
        # self.sim_target = sim_nograd(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=readout_noise_target)
        # self.sim_target.load_detector_properties(detector_props, pixel_layouts)

    def all_sim(self, tracks):
        TPB = 256
        BPG = max(ceil(tracks.shape[0] / TPB),1)
        print("Quenching electrons..." , end="")
        start_quenching = time()
        quenching.quench[BPG,TPB](tracks, consts.birks)
        end_quenching = time()
        print(f" {end_quenching-start_quenching:.2f} s")

        print("Drifting electrons...", end="")
        start_drifting = time()
        drifting.drift[BPG,TPB](tracks)
        end_drifting = time()
        print(f" {end_drifting-start_drifting:.2f} s")

        # # create a lookup table that maps between unique event ids and the segments in the file
        # tot_evids = np.unique(tracks[EVENT_SEPARATOR])
        # _, _, start_idx = np.intersect1d(tot_evids, tracks[EVENT_SEPARATOR], return_indices=True)
        # _, _, rev_idx = np.intersect1d(tot_evids, tracks[EVENT_SEPARATOR][::-1], return_indices=True)
        # end_idx = len(tracks[EVENT_SEPARATOR]) - 1 - rev_idx
        # track_ids = cp.array(np.arange(len(tracks)), dtype='i4')
        # # copy to device
        # track_ids = cp.asarray(np.arange(segment_ids.shape[0], dtype=int))

        # # create a lookup table for event timestamps
        # event_times = fee.gen_event_times(tot_evids.max()+1, 0)

        # rng_states = maybe_create_rng_states(1024*256, seed=0)
        # last_time = 0
        # # grab only tracks from current batch
        # track_subset = tracks[batch_mask]
        # if len(track_subset) == 0:
        #     continue
        # ievd = int(track_subset[0]['eventID'])
        # evt_tracks = track_subset
        # first_trk_id = np.argmax(batch_mask) # first track in batch

    #     for itrk in tqdm(range(0, evt_tracks.shape[0], BATCH_SIZE),
    #                     delay=1, desc='  Simulating event %i batches...' % ievd, leave=False, ncols=80):
    #         if itrk > 0:
    #             warnings.warn(f"Entered sub-batch loop, results may not be accurate! Consider reducing EVENT_BATCH_SIZE ({EVENT_BATCH_SIZE})")
                
    #         selected_tracks = evt_tracks[itrk:itrk+BATCH_SIZE]
    #         RangePush("event_id_map")
    #         event_ids = selected_tracks['eventID']
    #         unique_eventIDs = np.unique(event_ids)
    #         RangePop()

    #         # We find the pixels intersected by the projection of the tracks on
    #         # the anode plane using the Bresenham's algorithm. We also take into
    #         # account the neighboring pixels, due to the transverse diffusion of the charges.
    #         RangePush("pixels_from_track")
    #         max_radius = ceil(max(selected_tracks["tran_diff"])*5/detector.PIXEL_PITCH)

    #         TPB = 128
    #         BPG = max(ceil(selected_tracks.shape[0] / TPB),1)
    #         max_pixels = np.array([0])
    #         pixels_from_track.max_pixels[BPG,TPB](selected_tracks, max_pixels)

    #         # This formula tries to estimate the maximum number of pixels which can have
    #         # a current induced on them.
    #         max_neighboring_pixels = (2*max_radius+1)*max_pixels[0]+(1+2*max_radius)*max_radius*2

    #         active_pixels = cp.full((selected_tracks.shape[0], max_pixels[0]), -1, dtype=np.int32)
    #         neighboring_pixels = cp.full((selected_tracks.shape[0], max_neighboring_pixels), -1, dtype=np.int32)
    #         n_pixels_list = cp.zeros(shape=(selected_tracks.shape[0]))

    #         if not active_pixels.shape[1] or not neighboring_pixels.shape[1]:
    #             continue

    #         pixels_from_track.get_pixels[BPG,TPB](selected_tracks,
    #                                             active_pixels,
    #                                             neighboring_pixels,
    #                                             n_pixels_list,
    #                                             max_radius)
    #         RangePop()

    #         RangePush("unique_pix")
    #         shapes = neighboring_pixels.shape
    #         joined = neighboring_pixels.reshape(shapes[0] * shapes[1])
    #         unique_pix = cp.unique(joined)
    #         unique_pix = unique_pix[(unique_pix != -1)]
    #         RangePop()

    #         if not unique_pix.shape[0]:
    #             continue

    #         RangePush("time_intervals")
    #         # Here we find the longest signal in time and we store an array with the start in time of each track
    #         max_length = cp.array([0])
    #         track_starts = cp.empty(selected_tracks.shape[0])
    #         detsim.time_intervals[BPG,TPB](track_starts, max_length, selected_tracks)
    #         RangePop()

    #         RangePush("tracks_current")
    #         # Here we calculate the induced current on each pixel
    #         signals = cp.zeros((selected_tracks.shape[0],
    #                             neighboring_pixels.shape[1],
    #                             cp.asnumpy(max_length)[0]), dtype=np.float32)
    #         TPB = (1,1,64)
    #         BPG_X = max(ceil(signals.shape[0] / TPB[0]),1)
    #         BPG_Y = max(ceil(signals.shape[1] / TPB[1]),1)
    #         BPG_Z = max(ceil(signals.shape[2] / TPB[2]),1)
    #         BPG = (BPG_X, BPG_Y, BPG_Z)
    #         rng_states = maybe_create_rng_states(int(np.prod(TPB[:2]) * np.prod(BPG[:2])), seed=SEED+ievd+itrk, rng_states=rng_states)
    #         detsim.tracks_current_mc[BPG,TPB](signals, neighboring_pixels, selected_tracks, response, rng_states)
    #         RangePop()

    #         RangePush("pixel_index_map")
    #         # Here we create a map between tracks and index in the unique pixel array
    #         pixel_index_map = cp.full((selected_tracks.shape[0], neighboring_pixels.shape[1]), -1)
    #         for i_ in range(selected_tracks.shape[0]):
    #             compare = neighboring_pixels[i_, ..., cp.newaxis] == unique_pix
    #             indices = cp.where(compare)
    #             pixel_index_map[i_, indices[0]] = indices[1]
    #         RangePop()

    #         RangePush("track_pixel_map")
    #         # Mapping between unique pixel array and track array index
    #         track_pixel_map = cp.full((unique_pix.shape[0], detsim.MAX_TRACKS_PER_PIXEL), -1)
    #         TPB = 32
    #         BPG = max(ceil(unique_pix.shape[0] / TPB),1)
    #         detsim.get_track_pixel_map[BPG, TPB](track_pixel_map, unique_pix, neighboring_pixels)
    #         RangePop()

    #         RangePush("sum_pixels_signals")
    #         # Here we combine the induced current on the same pixels by different tracks
    #         TPB = (1,1,64)
    #         BPG_X = max(ceil(signals.shape[0] / TPB[0]),1)
    #         BPG_Y = max(ceil(signals.shape[1] / TPB[1]),1)
    #         BPG_Z = max(ceil(signals.shape[2] / TPB[2]),1)
    #         BPG = (BPG_X, BPG_Y, BPG_Z)
    #         pixels_signals = cp.zeros((len(unique_pix), len(detector.TIME_TICKS)))
    #         pixels_tracks_signals = cp.zeros((len(unique_pix),
    #                                         len(detector.TIME_TICKS),
    #                                         track_pixel_map.shape[1]))
    #         detsim.sum_pixel_signals[BPG,TPB](pixels_signals,
    #                                         signals,
    #                                         track_starts,
    #                                         pixel_index_map,
    #                                         track_pixel_map,
    #                                         pixels_tracks_signals)
    #         RangePop()

    #         RangePush("get_adc_values")
    #         # Here we simulate the electronics response (the self-triggering cycle) and the signal digitization
    #         time_ticks = cp.linspace(0, len(unique_eventIDs) * detector.TIME_INTERVAL[1], pixels_signals.shape[1]+1)
    #         integral_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
    #         adc_ticks_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
    #         current_fractions = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES, track_pixel_map.shape[1]))

    #         TPB = 128
    #         BPG = ceil(pixels_signals.shape[0] / TPB)
    #         rng_states = maybe_create_rng_states(int(TPB * BPG), seed=SEED+ievd+itrk, rng_states=rng_states)
    #         pixel_thresholds_lut.tpb = TPB
    #         pixel_thresholds_lut.bpg = BPG
    #         pixel_thresholds = pixel_thresholds_lut[unique_pix.ravel()].reshape(unique_pix.shape)

    #         fee.get_adc_values[BPG, TPB](pixels_signals,
    #                                     pixels_tracks_signals,
    #                                     time_ticks,
    #                                     integral_list,
    #                                     adc_ticks_list,
    #                                     0,
    #                                     rng_states,
    #                                     current_fractions,
    #                                     pixel_thresholds)

    #         adc_list = fee.digitize(integral_list)
    #         adc_event_ids = np.full(adc_list.shape, unique_eventIDs[0]) # FIXME: only works if looping on a single event
    #         RangePop()

    #         results_acc['event_id'].append(adc_event_ids)
    #         results_acc['adc_tot'].append(adc_list)
    #         results_acc['adc_tot_ticks'].append(adc_ticks_list)
    #         results_acc['unique_pix'].append(unique_pix)
    #         results_acc['current_fractions'].append(current_fractions)
    #         #track_pixel_map[track_pixel_map != -1] += first_trk_id + itrk
    #         track_pixel_map[track_pixel_map != -1] = track_ids[batch_mask][track_pixel_map[track_pixel_map != -1] + itrk]
    #         results_acc['track_pixel_map'].append(track_pixel_map)

    #     if len(results_acc['event_id']) > WRITE_BATCH_SIZE and len(np.concatenate(results_acc['event_id'], axis=0)) > 0:
    #         last_time = save_results(event_times, is_first_event=last_time==0, results=results_acc)
    #         results_acc = defaultdict(list)

    # # Always save results after last iteration
    # if len(results_acc['event_id']) >0 and len(np.concatenate(results_acc['event_id'], axis=0)) > 0:
    #     save_results(event_times, is_first_event=last_time==0, results=results_acc)

    # with h5py.File(output_filename, 'a') as output_file:
    #     if 'configs' in output_file.keys():
    #         output_file['configs'].attrs['pixel_layout'] = pixel_layout

    # print("Output saved in:", output_filename)
        
    def make_sim(self, dataloader, fixed_range, seed=1, epochs=300, iterations=None, shuffle=False, 
            save_freq=10, print_freq=1) -> None:
        # If explicit number of iterations, scale epochs accordingly
        if iterations is not None:
            epochs = iterations // len(dataloader) + 1

        # make a folder for the pixel target
        if os.path.exists('target_' + self.out_label):
            shutil.rmtree('target_' + self.out_label, ignore_errors=True)
        os.makedirs('target_' + self.out_label)

        for i, selected_tracks_bt_torch in enumerate(dataloader):
            event_id_map, unique_eventIDs = get_id_map(selected_tracks_bt_torch, self.track_dtypes.names, self.device)
            print(self.track_dtypes)
            print(selected_tracks_bt_torch.dtype)
            tracks = structured_from_torch(selected_tracks_bt_torch, self.track_dtypes)
            print(tracks)
            self.all_sim(tracks)
            # Get rid of the extra dimension and padding elements for the loaded data
            # selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
            # selected_tracks_bt_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("dx")] > 0]

            # for ev in unique_eventIDs:
            #     selected_tracks_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("eventID")] == ev]
            #     selected_tracks_torch = selected_tracks_torch.to(self.device)

            #     target, pix_target, ticks_list_targ = all_sim(self.sim_target, selected_tracks_torch, self.track_fields,
            #                                                     event_id_map, unique_eventIDs,
            #                                                     return_unique_pix=True)
            #     embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)

            #     torch.save(embed_target, 'target_' + self.out_label + '/batch' + str(i) + '_ev' + str(int(ev))+ '_target.pt')

def main(config):
    max_nbatch = config.max_nbatch

    dataset = TracksDataset(filename=config.input_file, ntrack=config.data_sz, max_nbatch=max_nbatch, seed=config.data_seed, random_ntrack=config.random_ntrack, 
                            track_len_sel=config.track_len_sel, track_z_bound=config.track_z_bound, max_batch_len=config.max_batch_len, print_input=config.print_input)

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
    param_fit = Simulation(dataset.get_track_dtypes(),
                            track_chunk=config.track_chunk, pixel_chunk=config.pixel_chunk,
                            detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                            readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                            readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                            norm_scheme=config.norm_scheme)
    param_fit.make_sim(tracks_dataloader, seed=config.seed, fixed_range=config.fixed_range)

    return 0, 'Simulation successful'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--batch_sz", dest="batch_sz", default=1, type=int,
                        help="Batch size for fitting (tracks).")
    parser.add_argument("--epochs", dest="epochs", default=100, type=int,
                        help="Number of epochs")
    parser.add_argument("--seed", dest="seed", default=2, type=int,
                        help="Random seed for target construction")
    parser.add_argument("--data_seed", dest="data_seed", default=3, type=int,
                        help="Random seed for data picking if not using the whole set")
    parser.add_argument("--data_sz", dest="data_sz", default=None, type=int,
                        help="Data size for fitting (number of tracks); input negative values to run on the whole dataset")
    parser.add_argument("--no-noise", dest="no_noise", default=False, action="store_true",
                        help="Flag to turn off readout noise (both target and guess)")
    parser.add_argument("--no-noise-target", dest="no_noise_target", default=False, action="store_true",
                        help="Flag to turn off readout noise (just target, guess has noise)")
    parser.add_argument("--no-noise-guess", dest="no_noise_guess", default=False, action="store_true",
                        help="Flag to turn off readout noise (just guess, target has noise)")
    parser.add_argument("--data_shuffle", dest="data_shuffle", default=False, action="store_true",
                        help="Flag of data shuffling")
    parser.add_argument("--random_ntrack", dest="random_ntrack", default=False, action="store_true",
                        help="Flag of whether sampling the tracks randomly or sequentially")
    parser.add_argument("--track_len_sel", dest="track_len_sel", default=2., type=float,
                        help="Track selection requirement on track length.")
    parser.add_argument("--track_z_bound", dest="track_z_bound", default=28., type=float,
                        help="Set z bound to keep healthy set of tracks")
    parser.add_argument("--fixed_range", dest="fixed_range", default=None, type=float,
                        help="Construct target by sampling in a certain range (fraction of nominal)")
    parser.add_argument("--norm_scheme", dest="norm_scheme", default="divide",
                        help="Normalization scheme to use for params. Right now, divide (by nom) and standard (subtract mean, div by variance)")
    parser.add_argument("--max_batch_len", dest="max_batch_len", default=None, type=float,
                        help="Max dx [cm] per batch. If passed, will add tracks to batch until overflow, splitting where needed")
    parser.add_argument("--max_nbatch", dest="max_nbatch", default=None, type=int,
                        help="Upper number of different batches taken from the data, given the max_batch_len. Overrides data_sz.")
    parser.add_argument("--print_input", dest="print_input", default=False, action="store_true",
                        help="print the event and track id per batch.")
    parser.add_argument("--shift-no-fit", dest="shift_no_fit", default=[], nargs="+", 
                        help="Set of params to shift in target sim without fitting them (robustness/separability check).")

    try:
        args = parser.parse_args()
        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    print(status_message)
    exit(retval)
