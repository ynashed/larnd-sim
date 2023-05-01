from .quenching_ep import quench
from .drifting_ep import drift
from .pixels_from_track_ep import pixels_from_track
from .detsim_ep import detsim
from .fee_ep import fee
import eagerpy as ep
from math import sqrt

#Wrapper derived class inheriting all simulation steps
class sim_with_grad(quench, drift, pixels_from_track, detsim, fee):
    def __init__(self, track_chunk=32, pixel_chunk=4, readout_noise=True, skip_pixels=False):
        quench.__init__(self)
        drift.__init__(self)
        pixels_from_track.__init__(self)
        detsim.__init__(self, track_chunk, pixel_chunk, skip_pixels)
        fee.__init__(self, readout_noise)

    def update_chunk_sizes(self, track_chunk, pixel_chunk):
        self.track_chunk = track_chunk
        self.pixel_chunk = pixel_chunk

    def estimate_peak_memory(self, tracks, fields):
        tracks_ep = ep.astensor(tracks)
        z_diff = ep.abs(tracks_ep[:, fields.index("z_end")] - tracks_ep[:, fields.index("z_start")])
        x_diff = ep.abs(tracks_ep[:, fields.index("x_end")] - tracks_ep[:, fields.index("x_start")])
        y_diff = ep.abs(tracks_ep[:, fields.index("y_end")] - tracks_ep[:, fields.index("y_start")])

        cond = x_diff.square() + y_diff.square() > 0

        cotan2 = ep.where(cond, z_diff.square()/(x_diff.square() + y_diff.square()), 0)

        pixel_diagonal = sqrt(self.pixel_pitch ** 2 + self.pixel_pitch ** 2)
        sigma_T = sqrt(((self.drift_length + 0.5)/self.vdrift)*2*self.tran_diff)
        sigma_L = sqrt(((self.drift_length + 0.5)/self.vdrift)*2*self.long_diff)
        impact_factor = max(pixel_diagonal, 10*sqrt(2)*sigma_T)

        time_max = ((z_diff.max() + 0.5)/self.vdrift + 2*self.time_padding)/self.t_sampling + 1
        t0_size = ep.maximum(30, ((1 + cotan2).sqrt()*impact_factor + 4*sigma_L).max()*4/self.t_sampling + 1)

        nb_elts = (time_max*t0_size*self.sampled_points*self.sampled_points).item()

        nb_bytes_per_elt = 128

        return nb_elts*nb_bytes_per_elt/1024/1024 + 200 #Returns in Mio with a safety margin
