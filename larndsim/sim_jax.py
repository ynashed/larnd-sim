import jax.numpy as jnp
from jax import jit, lax
import jax
import numpy as np
from numpy.lib import recfunctions as rfn
from flax import struct
from functools import partial

from larndsim.consts_jax import consts
from larndsim.detsim_jax import generate_electrons, get_pixels, id2pixel, accumulate_signals, current_mc
from larndsim.quenching_jax import quench
from larndsim.drifting_jax import drift
from larndsim.pixels_from_track_jax import get_pixel_coordinates
from larndsim.fee_jax import get_adc_values, digitize

def jax_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return jnp.array(tracks_np)

def load_data(fname):
    import h5py
    with h5py.File(fname, 'r') as f:
        tracks = np.array(f['segments'])

    x_start = np.copy(tracks['x_start'] )
    x_end = np.copy(tracks['x_end'])
    x = np.copy(tracks['x'])

    tracks['x_start'] = np.copy(tracks['z_start'])
    tracks['x_end'] = np.copy(tracks['z_end'])
    tracks['x'] = np.copy(tracks['z'])

    tracks['z_start'] = x_start
    tracks['z_end'] = x_end
    tracks['z'] = x

    selected_tracks = tracks
    dtype = selected_tracks.dtype
    return rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32), dtype

def chop_tracks(tracks, fields, precision=0.0001):
    def split_track(track, nsteps, length, direction, i):
        new_tracks = track.reshape(1, track.size).repeat(nsteps, axis=0)

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
    nsteps = np.maximum(np.ceil(length / precision), 1).astype(int)
    # step_size = length/nsteps
    new_tracks = np.vstack([split_track(tracks[i], nsteps[i], length[i], direction[i], i) for i in range(tracks.shape[0])])
    return new_tracks

def filter_tracks(tracks, fields):
    start = np.stack([tracks[:, fields.index("x_start")],
                        tracks[:, fields.index("y_start")],
                        tracks[:, fields.index("z_start")]], axis=1)
    end = np.stack([tracks[:, fields.index("x_end")],
                    tracks[:, fields.index("y_end")],
                    tracks[:, fields.index("z_end")]], axis=1)

    segment = end - start
    length = np.linalg.norm(segment, ord=2, axis=1)

    return tracks[length > 0]

def set_pixel_plane(params, tracks, fields):
    zMin = np.minimum(params.tpc_borders[:, 2, 1] - 2e-2, params.tpc_borders[:, 2, 0] - 2e-2)
    zMax = np.maximum(params.tpc_borders[:, 2, 1] + 2e-2, params.tpc_borders[:, 2, 0] + 2e-2)

    cond = tracks[:, fields.index("x")][..., None] >= params.tpc_borders[:, 0, 0][None, ...] - 2e-2
    cond = np.logical_and(tracks[:, fields.index("x")][..., None] <= params.tpc_borders[:, 0, 1][None, ...] + 2e-2, cond)
    cond = np.logical_and(tracks[:, fields.index("y")][..., None] >= params.tpc_borders[:, 1, 0][None, ...] - 2e-2, cond)
    cond = np.logical_and(tracks[:, fields.index("y")][..., None] <= params.tpc_borders[:, 1, 1][None, ...] + 2e-2, cond)
    cond = np.logical_and(tracks[:, fields.index("z")][..., None] >= zMin[None, ...], cond)
    cond = np.logical_and(tracks[:, fields.index("z")][..., None] <= zMax[None, ...], cond)

    mask = cond.sum(axis=-1) >= 1
    pixel_plane = cond.astype(int).argmax(axis=-1)
    tracks[:, fields.index('pixel_plane')] = pixel_plane
    return tracks

@partial(jit, static_argnames=['fields'])
def order_tracks_by_z(tracks, fields):
    #Modifies start and end in tracks so that z_start < z_end
    new_tracks = tracks.copy()
    cond = tracks[:, fields.index("z_start")] < tracks[:, fields.index("z_end")]

    new_tracks = new_tracks.at[:, fields.index("x_start")].set(lax.select(cond, tracks[:, fields.index("x_start")], tracks[:, fields.index("x_end")]))
    new_tracks = new_tracks.at[:, fields.index("x_end")].set(lax.select(cond, tracks[:, fields.index("x_end")], tracks[:, fields.index("x_start")]))
    new_tracks = new_tracks.at[:, fields.index("y_start")].set(lax.select(cond, tracks[:, fields.index("y_start")], tracks[:, fields.index("y_end")]))
    new_tracks = new_tracks.at[:, fields.index("y_end")].set(lax.select(cond, tracks[:, fields.index("y_end")], tracks[:, fields.index("y_start")]))
    new_tracks = new_tracks.at[:, fields.index("z_start")].set(lax.select(cond, tracks[:, fields.index("z_start")], tracks[:, fields.index("z_end")]))
    new_tracks = new_tracks.at[:, fields.index("z_end")].set(lax.select(cond, tracks[:, fields.index("z_end")], tracks[:, fields.index("z_start")]))

    return new_tracks

def loss(adcs, pIDs, ticks, adcs_ref, pIDs_ref, ticks_ref, fields):
    # return jnp.sqrt(jnp.sum((tracks[:, fields.index("n_electrons")] - tracks_ref[:, fields.index("n_electrons")])**2))
    
    unique_pixels = jnp.sort(jnp.unique(jnp.concatenate([pIDs, pIDs_ref])))
    nb_pixels = unique_pixels.shape[0]
    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs)

    pix_renumbering_ref = jnp.searchsorted(unique_pixels, pIDs_ref)

    signals = jnp.zeros((nb_pixels, adcs.shape[1]))
    signals = accumulate_signals(signals, adcs, pix_renumbering, jnp.zeros_like(pix_renumbering))
    signals = accumulate_signals(signals, -adcs_ref, pix_renumbering_ref, jnp.zeros_like(pix_renumbering_ref))

    adc_loss = jnp.sum(signals**2)

    # Add some penalty term for the time information also

    # signals = jnp.zeros((nb_pixels, adcs.shape[1]))
    # signals = accumulate_signals(signals, ticks, pix_renumbering, jnp.zeros_like(pix_renumbering))
    # signals = accumulate_signals(signals, -ticks_ref, pix_renumbering_ref, jnp.zeros_like(pix_renumbering_ref))
    # time_loss = jnp.sum(signals**2)
    time_loss = 0

    aux = {
        'adc_loss': adc_loss,
        'time_loss': time_loss
    }

    return adc_loss + time_loss, aux

def simulate(params, tracks, fields, rngkey = 0):
    new_tracks = quench(params, tracks, 2, fields)
    new_tracks = drift(params, new_tracks, fields)
    electrons = generate_electrons(new_tracks, fields, rngkey)
    pIDs = get_pixels(params, electrons, fields)
    
    xpitch, ypitch, plane, eid = id2pixel(params, pIDs)
    
    pixels_coord = get_pixel_coordinates(params, xpitch, ypitch, plane)
    t0, signals = current_mc(params, electrons, pixels_coord, fields)
    unique_pixels = jnp.sort(jnp.unique(pIDs))
    npixels = unique_pixels.shape[0]

    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs)

    nticks_wf = int(params.time_window/params.t_sampling)
    wfs = jnp.zeros((npixels, nticks_wf))

    
    start_ticks = (t0/params.t_sampling + 0.5).astype(int)
    earliest_tick = jnp.min(start_ticks)

    wfs = accumulate_signals(wfs, signals, pix_renumbering, start_ticks - earliest_tick)
    # return signals, pix_renumbering, wfs, start_ticks, unique_pixels
    integral, ticks = get_adc_values(params, wfs)
    adcs = digitize(params, integral)
    # return wfs, unique_pixels
    return adcs, unique_pixels, ticks

def params_loss(params, ref, pixels_ref, ticks_ref, tracks, fields, rngkey=0):
    adcs, pixels, ticks = simulate(params, tracks, fields, rngkey)
    return loss(adcs, pixels, ticks, ref, pixels_ref, ticks_ref, fields)

def update_params(params, params_init, grads, to_propagate, lr):
    modified_values = {}
    for key in to_propagate:
        modified_values[key] = getattr(params, key) - getattr(params_init, key)**2 * getattr(grads, key)*lr
    return params.replace(**modified_values)

def prepare_tracks(params, tracks_file):
    
    tracks, dtype = load_data(tracks_file)
    fields = dtype.names

    tracks = set_pixel_plane(params, tracks, fields)
    original_tracks = tracks.copy()
    tracks = chop_tracks(tracks, fields)
    tracks = tracks[:200000]
    tracks = jnp.array(tracks)

    # tracks = order_tracks_by_z(tracks, fields)
    # tracks = filter_tracks(tracks, fields)

    return tracks, fields, original_tracks

@struct.dataclass
class Params:
    eField: float = struct.field(pytree_node=False)
    Ab: float #= struct.field(pytree_node=False)
    kb: float
    lifetime: float
    vdrift: float = struct.field(pytree_node=False)
    long_diff: float = struct.field(pytree_node=False)
    tran_diff: float = struct.field(pytree_node=False)
    tpc_borders: jax.Array = struct.field(pytree_node=False)
    box: int = struct.field(pytree_node=False)
    birks: int = struct.field(pytree_node=False)
    lArDensity: float = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)
    beta: float = struct.field(pytree_node=False)
    MeVToElectrons: float = struct.field(pytree_node=False)
    pixel_pitch: float = struct.field(pytree_node=False)
    n_pixels: tuple = struct.field(pytree_node=False)
    max_radius: int = struct.field(pytree_node=False)
    max_active_pixels: int = struct.field(pytree_node=False)
    drift_length: float = struct.field(pytree_node=False)
    t_sampling: float = struct.field(pytree_node=False)
    time_interval: float = struct.field(pytree_node=False)
    time_padding: float = struct.field(pytree_node=False)
    min_step_size: float = struct.field(pytree_node=False)
    time_max: float = struct.field(pytree_node=False)
    time_window: float = struct.field(pytree_node=False)
    e_charge: float = struct.field(pytree_node=False)
    #: Maximum number of ADC values stored per pixel
    MAX_ADC_VALUES: int = struct.field(pytree_node=False)
    #: Discrimination threshold
    DISCRIMINATION_THRESHOLD: float = struct.field(pytree_node=False)
    #: ADC hold delay in clock cycles
    ADC_HOLD_DELAY: int = struct.field(pytree_node=False)
    #: Clock cycle time in :math:`\mu s`
    CLOCK_CYCLE: float = struct.field(pytree_node=False)
    #: Front-end gain in :math:`mV/ke-`
    GAIN: float = struct.field(pytree_node=False)
    #: Common-mode voltage in :math:`mV`
    V_CM: float = struct.field(pytree_node=False)
    #: Reference voltage in :math:`mV`
    V_REF: float = struct.field(pytree_node=False)
    #: Pedestal voltage in :math:`mV`
    V_PEDESTAL: float = struct.field(pytree_node=False)
    #: Number of ADC counts
    ADC_COUNTS: int = struct.field(pytree_node=False)
    # if readout_noise:
        #: Reset noise in e-
        # self.RESET_NOISE_CHARGE = 900
        # #: Uncorrelated noise in e-
        # self.UNCORRELATED_NOISE_CHARGE = 500
    # else:
    RESET_NOISE_CHARGE: float = struct.field(pytree_node=False)
    UNCORRELATED_NOISE_CHARGE: float = struct.field(pytree_node=False)

def load_params() -> Params:
    csts = consts()
    csts.load_detector_properties("/home/pgranger/larnd-sim/jit_version/larnd-sim/larndsim/detector_properties/module0.yaml",
                                "/home/pgranger/larnd-sim/jit_version/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml")
    params = {
    "eField": 0.50,
    "Ab": 0.8,
    "kb": 0.0486,
    "vdrift": 0.1648,
    "lifetime": 2.2e3,
    "long_diff": 4.0e-6,
    "tran_diff": 8.8e-6,
    "tpc_borders": jnp.asarray(csts.tpc_borders),
    "box": 1,
    "birks": 2,
    "lArDensity": 1.38,
    "alpha": 0.93,
    "beta": 0.207,
    "MeVToElectrons": 4.237e+04,
    "pixel_pitch": csts.pixel_pitch,
    "n_pixels": csts.n_pixels,
    "drift_length": csts.drift_length,
    # "t_sampling": csts.t_sampling,
    "t_sampling": csts.t_sampling,
    "time_interval": csts.time_interval,
    "time_padding": csts.time_padding,
    "max_active_pixels": 0,
    "max_radius": 0,
    "min_step_size": 0.001, #cm
    "time_max": 0,
    "time_window": 189.1, #us,
    "e_charge": 1.602e-19,
    "MAX_ADC_VALUES": 10,
    "DISCRIMINATION_THRESHOLD": 7e3*1.602e-19,
    "ADC_HOLD_DELAY": 15,
    "CLOCK_CYCLE": 0.1,
    "GAIN": 4e-3,
    "V_CM": 288,
    "V_REF": 1300,
    "V_PEDESTAL": 580,
    "ADC_COUNTS": 2**8,
    "RESET_NOISE_CHARGE": 0,
    "UNCORRELATED_NOISE_CHARGE": 0,
    }

    return Params(**params)