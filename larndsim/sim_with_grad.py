from .quenching_ep import quench
from .drifting_ep import drift
from .pixels_from_track_ep import pixels_from_track
from .detsim_ep import detsim
from .fee_ep import fee

#Wrapper derived class inheriting all simulation steps
class sim_with_grad(quench, drift, pixels_from_track, detsim, fee):
    def __init__(self):
        super().__init__()
