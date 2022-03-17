import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib import recfunctions as rfn

def torch_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return torch.from_numpy(tracks_np).float()

def structured_from_torch(tracks_torch, dtype):
    return rfn.unstructured_to_structured(tracks_torch.cpu().numpy(), dtype=dtype)

class TracksDataset(Dataset):
    def __init__(self, filename, swap_xz=True):

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

        tracks = tracks[200:250]
        self.track_fields = tracks.dtype.names
        #self.tracks = torch_from_structured(tracks)

        # flat index for eventID and trackID
        self.index = []
        num_tracks = 0
        all_tracks = []
        all_events = np.unique(tracks['eventID'])
        for ev in all_events:
            track_set = np.unique(tracks[tracks['eventID'] == ev]['trackID'])
            for trk in track_set:
                # basic track selection
                trk_msk = (tracks['eventID'] == ev) & (tracks['trackID'] == trk)
                if max(tracks[trk_msk]['z']) - min(tracks[trk_msk]['z']) > 3:
                    # add event, track index to the list
                    self.index.append([ev, trk])
                    all_tracks.append(torch_from_structured(tracks[trk_msk]))
        self.tracks = torch.nn.utils.rnn.pad_sequence(all_tracks, batch_first=True, padding_value = -99) 
        print("trainning track set [ev, trk]: ", self.index)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]
        
    def get_track_fields(self):
        return self.track_fields

