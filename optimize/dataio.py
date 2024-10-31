import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib import recfunctions as rfn
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def torch_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return torch.from_numpy(tracks_np).float()

def structured_from_torch(tracks_torch, dtype):
    return rfn.unstructured_to_structured(tracks_torch.cpu().numpy(), dtype=dtype)

def chop_tracks(tracks, fields, precision=0.001):
    def split_track(track, nsteps, length, direction, i):
        new_tracks = track.reshape(1, track.shape[0]).repeat(nsteps, axis=0)

        new_tracks[:, fields.index("dE")] = new_tracks[:, fields.index("dE")]*precision/(length+1e-10)
        steps = np.arange(0, nsteps)

        new_tracks[:, fields.index("x_start")] = track[fields.index("x_start")] + steps*precision*direction[0]
        new_tracks[:, fields.index("y_start")] = track[fields.index("y_start")] + steps*precision*direction[1]
        new_tracks[:, fields.index("z_start")] = track[fields.index("z_start")] + steps*precision*direction[2]

        new_tracks[:, fields.index("x_end")] = track[fields.index("x_start")] + precision*(steps + 1)*direction[0]
        new_tracks[:, fields.index("y_end")] = track[fields.index("y_start")] + precision*(steps + 1)*direction[1]
        new_tracks[:, fields.index("z_end")] = track[fields.index("z_start")] + precision*(steps + 1)*direction[2]
        new_tracks[:, fields.index("dx")] = precision

        #Correcting the last track bit
        new_tracks[-1, fields.index("x_end")] = track[fields.index("x_end")]
        new_tracks[-1, fields.index("y_end")] = track[fields.index("y_end")]
        new_tracks[-1, fields.index("z_end")] = track[fields.index("z_end")]
        new_tracks[-1, fields.index("dE")] = track[fields.index("dE")]*(1 - precision*(nsteps - 1)/(length + 1e-10))
        new_tracks[-1, fields.index("dx")] = length - precision*(nsteps - 1)

        #Finally computing the middle point once everything is ok
        new_tracks[:, fields.index("x")] = 0.5*(new_tracks[:, fields.index("x_start")] + new_tracks[:, fields.index("x_end")])
        new_tracks[:, fields.index("y")] = 0.5*(new_tracks[:, fields.index("y_start")] + new_tracks[:, fields.index("y_end")])
        new_tracks[:, fields.index("z")] = 0.5*(new_tracks[:, fields.index("z_start")] + new_tracks[:, fields.index("z_end")])

        # orig_track = np.full((new_tracks.shape[0], 1), i)
        # new_tracks = np.hstack([new_tracks, orig_track])
        return new_tracks
    
    tracks = tracks.numpy()
    
    start = np.stack([tracks[:, fields.index("x_start")],
                        tracks[:, fields.index("y_start")],
                        tracks[:, fields.index("z_start")]], axis=1)
    end = np.stack([tracks[:, fields.index("x_end")],
                    tracks[:, fields.index("y_end")],
                    tracks[:, fields.index("z_end")]], axis=1)

    segment = end - start
    length = np.sqrt(np.sum(segment**2, axis=1, keepdims=True))
    eps = 1e-10
    direction = segment / (length + eps)
    nsteps = np.maximum(np.ceil(length / precision), 1).astype(int).flatten()
    # step_size = length/nsteps
    new_tracks = np.vstack([split_track(tracks[i], nsteps[i], length[i], direction[i], i) for i in range(tracks.shape[0])])
    return new_tracks

class TracksDataset(Dataset):
    def __init__(self, filename, ntrack, max_nbatch=None, swap_xz=True, seed=3, random_ntrack=False, track_len_sel=2., 
                 max_abs_costheta_sel=0.966, min_abs_segz_sel=15., track_z_bound=28., max_batch_len=None, print_input=False,
                 chopped=True, pad=True):

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

        

        if not 't0' in tracks.dtype.names:
            tracks = rfn.append_fields(tracks, 't0', np.zeros(tracks.shape[0]), usemask=False)
        
        self.track_fields = tracks.dtype.names

        # flat index for all reasonable track [eventID, trackID] 
        index = []
        all_tracks = []
        for ev in np.unique(tracks['eventID']):
            track_set = np.unique(tracks[tracks['eventID'] == ev]['trackID'])
            for trk in track_set:
                trk_msk = (tracks['eventID'] == ev) & (tracks['trackID'] == trk)
                xd = tracks[trk_msk]['x_start'][0] - tracks[trk_msk]['x_end'][-1]
                yd = tracks[trk_msk]['y_start'][0] - tracks[trk_msk]['y_end'][-1]
                zd = tracks[trk_msk]['z_start'][0] - tracks[trk_msk]['z_end'][-1]
                z_dir = [0,0,1]
                trk_dir = [xd, yd, zd]
                if np.sum(tracks[trk_msk]['dx']) > track_len_sel:
                    cos_theta = abs(np.dot(trk_dir, z_dir))/ np.linalg.norm(trk_dir)
                #TODO once we enter the end game, this track selection requirement needs to be more accessible.
                # For now, we keep it as it is to take consistent data among developers
                if np.sum(tracks[trk_msk]['dx']) > track_len_sel and max(abs(tracks[trk_msk]['z'])) < track_z_bound and abs(cos_theta) < max_abs_costheta_sel:
                    index.append([ev, trk])
                    all_tracks.append(torch_from_structured(tracks[trk_msk][abs(tracks[trk_msk]['z']) > min_abs_segz_sel]))

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
            logger.info(f"training set [ev, trk]: {fit_index}")
      
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
                                logger.info(f"~ [batch ID]: {len(batches)}")
                                logger.info(f"  batch length: {tot_length - segment[self.track_fields.index('dx')]}")
                                logger.info(f"  event IDs: {ev_here}")
                                logger.info(f"  track IDs: {trk_here}")
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
            
            if chopped:
                fit_tracks = [torch.tensor(chop_tracks(batch, self.track_fields)) for batch in batches]
            else:
                fit_tracks = batches
            logger.info(f"-- The used data includes a total track length of {tot_data_length} cm.")
            logger.info(f"-- The maximum batch track length is {max_batch_len} cm.")
            logger.info(f"-- There are {len(batches)} different batches in total.")
        if pad:
            self.tracks = torch.nn.utils.rnn.pad_sequence(fit_tracks, batch_first=True, padding_value = 0)
        else:
            self.tracks = fit_tracks

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx].float()
        
    def get_track_fields(self):
        return self.track_fields

