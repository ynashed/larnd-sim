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
    def __init__(self, filename, ntrack, swap_xz=True, seed=3):

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
                index.append([ev, trk])
                all_tracks.append(torch_from_structured(tracks[trk_msk]))

        # all fit with a sub-set of tracks
        fit_index = []
        fit_tracks = []
        if ntrack >= len(index):
            self.tracks = torch.nn.utils.rnn.pad_sequence(all_tracks, batch_first=True, padding_value = -99) 
            fit_index = index
        else:
            # if the information of track index is uninteresting, then the next line + pad_sequence is enough
            # fit_tracks = random.sample(all_tracks, ntrack)
            random.seed(seed)
            list_rand = random.sample(range(len(index)), ntrack)
            for i_rand in list_rand:
                fit_index.append(index[i_rand])
                fit_tracks.append(all_tracks[i_rand])
            self.tracks = torch.nn.utils.rnn.pad_sequence(fit_tracks, batch_first=True, padding_value = -99) 
        
        #self.tracks = torch.nn.utils.rnn.pad_sequence(all_tracks, batch_first=True, padding_value = -99) 
        print("training set [ev, trk]: ", fit_index)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]
        
    def get_track_fields(self):
        return self.track_fields

