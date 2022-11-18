import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib import recfunctions as rfn
import random

def torch_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return torch.from_numpy(tracks_np).float()

def structured_from_torch(tracks_torch, dtype):
    return rfn.unstructured_to_structured(tracks_torch.cpu().numpy(), dtype=dtype)

class TracksDataset(Dataset):
    def __init__(self, filename, ntrack, max_nbatch=None, swap_xz=True, seed=3, random_ntrack=False, track_len_sel=2., 
                 track_z_bound=28., max_batch_len=None, print_input=False):

        with h5py.File(filename, 'r') as f:
            tracks = np.array(f['segments'])

        if swap_xz:
            x_start = np.copy(tracks['x_start'] )
            x_end = np.copy(tracks['x_end'])
            x = np.copy(tracks['x'])

            tracks['x_start'] = np.copy(tracks['z_start'])
            tracks['x_end'] = np.copy(tracks['z_end'])
            tracks['x'] = np.copy(tracks['z'])

            tracks['z_start'] = x_start
            tracks['z_end'] = x_end
            tracks['z'] = x

        self.track_fields = tracks.dtype.names

        # flat index for all reasonable track [eventID, trackID] 
        index = []
        all_tracks = []
        for ev in np.unique(tracks['eventID']):
            track_set = np.unique(tracks[tracks['eventID'] == ev]['trackID'])
            for trk in track_set:
                trk_msk = (tracks['eventID'] == ev) & (tracks['trackID'] == trk)
                #TODO once we enter the end game, this track selection requirement needs to be more accessible.
                # For now, we keep it as it is to take consistent data among developers
                if np.sum(tracks[trk_msk]['dx']) > track_len_sel and max(abs(tracks[trk_msk]['z'])) < track_z_bound:
                    index.append([ev, trk])
                    all_tracks.append(torch_from_structured(tracks[trk_msk]))

        # all fit with a sub-set of tracks
        fit_index = []
        fit_tracks = []
        random.seed(seed)
        if ntrack is None or ntrack >= len(index) or ntrack <= 0:
            if random_ntrack:
                random.shuffle(all_tracks)
            fit_tracks = all_tracks
            fit_index = index
        else:
            # if the information of track index is uninteresting, then the next line + pad_sequence is enough
            # fit_tracks = random.sample(all_tracks, ntrack)
            if random_ntrack:
                list_rand = random.sample(range(len(index)), ntrack)
            else:
                list_rand = np.arange(ntrack)
                
            for i_rand in list_rand:
                fit_index.append(index[i_rand])
                fit_tracks.append(all_tracks[i_rand])

        if print_input:
            print("training set [ev, trk]: ", fit_index)
      
        if max_batch_len is not None:
            batches = []
            batch_here = []
            ev_here = []
            trk_here = []
            tot_length = 0
            tot_data_length = 0
            done_track_looping = False
            for track in fit_tracks:
                for segment in track:
                    if segment[self.track_fields.index("dx")] > max_batch_len:
                        continue
                    tot_length+=segment[self.track_fields.index("dx")]
                    if tot_length < max_batch_len:
                        batch_here.append(segment)
                        ev_here.append(segment[[self.track_fields.index("eventID")]])
                        trk_here.append(segment[[self.track_fields.index("trackID")]])
                    else:
                       
                        if len(batch_here) > 0:
                            batches.append(torch.stack(batch_here))
                            tot_data_length += tot_length - segment[self.track_fields.index("dx")]
                            if print_input:
                                print("~ [batch ID]: ", len(batches))
                                print("  batch length: ", tot_length - segment[self.track_fields.index("dx")])
                                print("  event IDs: ", ev_here)
                                print("  track IDs: ", trk_here)
                        batch_here = []
                        ev_here = []
                        trk_here = []
                        tot_length = 0
                        if max_nbatch is not None and len(batches) >= max_nbatch and max_nbatch > 0: 
                            done_track_looping = True
                            break
                        batch_here.append(segment)
                        ev_here.append(segment[[self.track_fields.index("eventID")]])
                        trk_here.append(segment[[self.track_fields.index("trackID")]])
                        tot_length+=segment[self.track_fields.index("dx")]
                if done_track_looping:
                    break
            if len(batch_here) > 0:
                batches.append(torch.stack(batch_here))
                tot_data_length += tot_length
            
            fit_tracks = batches

            print(f"-- The used data includes a total track length of {tot_data_length} cm.")
            print(f"-- The maximum batch track length is {max_batch_len} cm.")
            print(f"-- There are {len(batches)} different batches in total.")

        self.tracks = torch.nn.utils.rnn.pad_sequence(fit_tracks, batch_first=True, padding_value = -99) 

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx].double()
        
    def get_track_fields(self):
        return self.track_fields

