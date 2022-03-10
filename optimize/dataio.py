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

        tracks = tracks[:300]
        self.track_fields = tracks.dtype.names
        self.tracks = torch_from_structured(tracks)

        # flat index for eventID and trackID
        self.index = []
        #track_arr = np.([])
        num_tracks = 0
        all_events = np.unique(tracks['eventID'])
        for ev in all_events:
            track_set = np.unique(tracks[tracks['eventID'] == ev]['trackID'])
            for trk in track_set:
                # basic track selection
                trk_msk = (tracks['eventID'] == ev) & (tracks['trackID'] == trk)
                if max(tracks[trk_msk]['z']) - min(tracks[trk_msk]['z']) > 30:
                    #print("here: ", tracks[trk_msk])
                    # add event, track index to the list
                    self.index.append([ev, trk])
            	    #track_list.append(torch_from_structured(tracks[trk_msk]))
                    #if num_tracks > 0:
                    #  self.tracks = torch.cat((self.tracks, torch_from_structured(tracks[trk_msk])))
                    #else:
                    #  self.tracks = torch_from_structured(tracks[trk_msk])
                    #num_tracks += 1
        
        #self.tracks = torch.rand(len(self.index))
        
        #for i_trk in range(len(self.index)):
        #  trk_msk = (tracks['eventID'] == self.index[i_trk][0]) & (tracks['trackID'] == self.index[i_trk][1])
        #  print(tracks[trk_msk])
        #  print(torch_from_structured(tracks[trk_msk]))
        #  print(torch_from_structured(tracks[trk_msk])[:, self.track_fields.index('dx')])
        #  #self.tracks[i_trk] = 2
        #  print("this track: ", torch_from_structured(tracks[trk_msk]))
        #  this_track = torch_from_structured(tracks[trk_msk])
        #  this_track = this_track[None, :]
        #  print("this_track: ", this_track)
        #  if i_trk > 0:
        #    self.tracks = torch.cat((self.tracks, this_track))
        #    #self.tracks = torch.stack((self.tracks, torch_from_structured(tracks[trk_msk])))
        #  else:
        #    self.tracks = this_track
        #    #self.tracks = torch_from_structured(tracks[trk_msk])
        #    

        #  #self.tracks[i_trk] = torch_from_structured(tracks[trk_msk])
        ##print("tracklist:", track_list)
        ##self.tracks = torch.Tensor(track_list)
        #print("ntrk: ", len(self.index))
        #print("all track: ", self.tracks)
         
        print(self.index)

    def __len__(self):
        return len(self.tracks)

    #def __getitem__(self):
    def __getitem__(self, idx):
        print("idx: ", idx)
        idx_mask = (self.tracks[:, self.track_fields.index('eventID')] == self.index[idx][0]) & (self.tracks[:, self.track_fields.index('trackID')] == self.index[idx][1])
        return self.tracks[idx_mask]

    def get_track_fields(self):
        return self.track_fields

#    def collate_batch(self, idx):
#        print("ev id: ", self.index[idx][0])
#        print("trk id: ",self.index[idx][1])
#        idx_mask = (self.tracks[:, self.track_fields.index('eventID')] == self.index[idx][0]) & (self.tracks[:, self.track_fields.index('trackID')] == self.index[idx][1])
#        return self.tracks[idx_mask] 
#
#    def pin_memory(self):
#        print("pin_memory")
#        self.tracks = self.tracks.pin_memory()
#        return self
