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

        self.track_fields = tracks.dtype.names
        self.tracks = torch_from_structured(tracks)

        # self.index = {}
        # all_events = np.unique(self.tracks['eventID'])
        # for ev in all_events:
        #     track_set = np.unique(self.tracks[self.tracks['eventID'] == ev]['trackID'])
        #     self.index[ev] = track_set


    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]

    def get_track_fields(self):
        return self.track_fields