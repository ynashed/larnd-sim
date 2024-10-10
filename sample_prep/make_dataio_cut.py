#!/usr/bin/env python3

import h5py
import torch
import numpy as np
from numpy.lib import recfunctions as rfn
import os, sys
from tqdm import tqdm
from glob import glob
import argparse


# used in dataio
def torch_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    #return torch.from_numpy(tracks_np).float() # changed in dataio, cant store torch arrays
    return np.array(tracks_np)

def dataio_cut(filename, out_name, dEdx_threshold, is_max, 
               swap_xz, track_len_sel, max_abs_costheta_sel, min_abs_segz_sel, track_z_bound):
    
    print(f"Loading tracks from {filename}...")
    with h5py.File(filename, 'r') as f:
        tracks = np.array(f['segments'])
    num_tot_tracks = len(tracks)
    print(f"There are {num_tot_tracks} tracks")
    
    if swap_xz:
        print("Swapping x & z...")
        x_start = np.copy(tracks['x_start'] )
        x_end = np.copy(tracks['x_end'])
        x = np.copy(tracks['x'])

        tracks['x_start'] = np.copy(tracks['z_start'])
        tracks['x_end'] = np.copy(tracks['z_end'])
        tracks['x'] = np.copy(tracks['z'])

        tracks['z_start'] = x_start
        tracks['z_end'] = x_end
        tracks['z'] = x

    # flat index for all reasonable track [eventID, trackID] 
    add = "<" if is_max else ">"
    if dEdx_threshold: # is not None
        print(f"Cutting tracks w/ dEdx {add} {dEdx_threshold}...")
    else:
        print(f"Cutting tracks...")
    # make datasets to store in h5
    index = []
    all_tracks = np.empty(shape=(1,len(tracks.dtype.names)), dtype='float64')
    cut_tracks = np.empty(shape=(1), dtype=tracks.dtype) ##[track1, track2]
    events = np.unique(tracks['eventID'])
    num_tracks_added = 0
    with tqdm(total = len(events)) as pbar:
        for count, ev in enumerate(events):
            track_set = np.unique(tracks[tracks['eventID'] == ev]['trackID'])
            for trk in track_set:
                trk_msk = (tracks['eventID'] == ev) & (tracks['trackID'] == trk)
                track = tracks[trk_msk]
                xd = track['x_start'][0] - track['x_end'][-1]
                yd = track['y_start'][0] - track['y_end'][-1]
                zd = track['z_start'][0] - track['z_end'][-1]
                z_dir = [0,0,1]
                trk_dir = [xd, yd, zd]
                # selection criteria: track length, track direction not in z, max z comp not too high-->
                #                     now select only segments in track that arent nuclei and above min z
                if np.sum(track['dx']) > track_len_sel:
                    cos_theta = abs(np.dot(trk_dir, z_dir))/ np.linalg.norm(trk_dir)
                    if max(abs(track['z'])) < track_z_bound and abs(cos_theta) < max_abs_costheta_sel:
                        msk = np.logical_and(abs(track['z']) > min_abs_segz_sel, track['pdgId'] < 1e6)
                        num_new_tracks = len(track[msk])
                        if num_new_tracks > 0:  # continue if masked tracks are nonzero
                            # ENERGY threshold CUT
                            if dEdx_threshold != None:
                                mean_energy = np.mean(track[msk]['dEdx'])
                                if (mean_energy > dEdx_threshold and not is_max) or (mean_energy < dEdx_threshold and is_max):
                                    num_tracks_added += num_new_tracks
                                    index.append([ev, trk])
                                    cut_tracks = np.append(cut_tracks, track[msk])
                                    all_tracks = np.append(all_tracks, torch_from_structured(track[msk]), 0)
                            else:
                                num_tracks_added += num_new_tracks
                                index.append([ev, trk])
                                cut_tracks = np.append(cut_tracks, track[msk])
                                all_tracks = np.append(all_tracks, torch_from_structured(track[msk]), 0)
            print(f"B-) Number of cut tracks: {num_tracks_added}")
            pbar.update(1)
    # now generate new h5py file with said datasets
    if out_name == None:
        out_name = filename.split('/')[-1].split('.h5')[0]+(str(dEdx_threshold) if dEdx_threshold > 0 else '')
    f = h5py.File(f'{out_name}.h5','w')
    f.create_dataset('segments', data=cut_tracks)   # all cut segments
    f.create_dataset('index', data=index)           # indices of said segments
    f.create_dataset('all_tracks', data=all_tracks) # torch formatted data
    f.close()
    print(f"Saved cut data ({num_tracks_added}/{num_tot_tracks}) to {out_name}.h5 in {os.getcwd()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="filename", help="Filename to make data cut from")
    parser.add_argument("--out", dest="out_name", default = None,
                        help="Name of output file for making dataio")
    parser.add_argument("--threshold", dest="threshold", default = None, type=float,
                        help="Threshold for minimum/maximum avg dEdx value to cut from")
    parser.add_argument("--max", dest="is_max", default = False, action="store_true",
                        help="default threshold is the min avg dedx, change to the max avg dedx")     
    parser.add_argument("--swap_xz", dest="swap_xz", default=True, action="store_false",
                        help="Swap x and z axes in data")             
    parser.add_argument("--track_len_sel", dest="track_len_sel", default=2., type=float,
                        help="Track selection requirement on track length.")
    parser.add_argument("--max_abs_costheta_sel", dest="max_abs_costheta_sel", default=0.966, type=float,
                        help="Theta is the angle of track wrt to the z axis. Remove tracks which are very colinear with z.")
    parser.add_argument("--min_abs_segz_sel", dest="min_abs_segz_sel", default=15., type=float,
                        help="Remove track segments that are close to the cathode.")
    parser.add_argument("--track_z_bound", dest="track_z_bound", default=28., type=float,
                        help="Set z bound to keep healthy set of tracks")
    args = parser.parse_args()
    dataio_cut(args.filename, args.out_name, args.threshold, args.is_max, args.swap_xz,
               args.track_len_sel, args.max_abs_costheta_sel, args.min_abs_segz_sel, args.track_z_bound)
    