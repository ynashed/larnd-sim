import jax.numpy as jnp
from jax import jit, lax
import numpy as np
from numpy.lib import recfunctions as rfn
from functools import partial
import logging
from jax.experimental import checkify

# from larndsim.consts_jax import consts
from larndsim.detsim_jax import generate_electrons, get_pixels, id2pixel, accumulate_signals, accumulate_signals_parametrized, current_lut, get_pixel_coordinates, current_mc
from larndsim.quenching_jax import quench
from larndsim.drifting_jax import drift
from larndsim.fee_jax import get_adc_values, digitize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

size_history_dict = {}

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

def loss(adcs, pIDs, ticks, adcs_ref, pIDs_ref, ticks_ref, fields):
    # return jnp.sqrt(jnp.sum((tracks[:, fields.index("n_electrons")] - tracks_ref[:, fields.index("n_electrons")])**2))
    #TODO: Put back something that is actually good here!
    all_pixels = jnp.concatenate([pIDs, pIDs_ref])
    padded_size = pad_size(jnp.unique(all_pixels).shape[0], "unique_pixels")
    unique_pixels = jnp.sort(jnp.unique(all_pixels, size=padded_size, fill_value=-1))
    nb_pixels = unique_pixels.shape[0]
    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs)

    pix_renumbering_ref = jnp.searchsorted(unique_pixels, pIDs_ref)

    signals = jnp.zeros((nb_pixels, adcs.shape[1]))
    signals = signals.at[pix_renumbering, :].add(adcs)
    signals = signals.at[pix_renumbering_ref, :].add(-adcs_ref)

    # signals = accumulate_signals(signals, adcs, pix_renumbering, jnp.zeros_like(pix_renumbering))
    # indices = jnp.expand_dims(pix_renumbering, axis=1) * signals.shape[1] + jnp.arange(signals.shape[1])
    # Flatten the indices
    # flat_indices = jnp.ravel(indices)

    # Update wfs with accumulated signals
    # wfs = wfs.ravel()
    # wfs = wfs.at[(flat_indices,)].add(adcs.ravel())
    # indices = jnp.expand_dims(pix_renumbering_ref, axis=1) * signals.shape[1] + jnp.arange(signals.shape[1])
    # flat_indices = jnp.ravel(indices)
    # wfs = wfs.at[(flat_indices,)].add(adcs.ravel())

    # signals = accumulate_signals(signals, -adcs_ref, pix_renumbering_ref, jnp.zeros_like(pix_renumbering_ref))
    
    adc_loss = jnp.sum(signals**2)

    # Add some penalty term for the time information also

    signals = jnp.zeros((nb_pixels, adcs.shape[1]))
    signals = accumulate_signals_parametrized(signals, ticks, pix_renumbering, jnp.zeros_like(pix_renumbering))
    signals = accumulate_signals_parametrized(signals, -ticks_ref, pix_renumbering_ref, jnp.zeros_like(pix_renumbering_ref))
    time_loss = jnp.sum(signals**2)
    time_loss = 0

    aux = {
        'adc_loss': adc_loss,
        'time_loss': time_loss,
        'ticks': ticks,
        'adcs': adcs,
        'pixels': pIDs
    }

    return adc_loss + time_loss, aux

def pad_size(cur_size, tag):
    global size_history_dict

    if tag not in size_history_dict:
        size_history_dict[tag] = []
    size_history = size_history_dict[tag]

    pad_threshold = 0.05
    #If an input with this shape has already been used, we are fine
    if cur_size in size_history:
        logger.debug(f"Input size {cur_size} already existing.")
        return cur_size
    #Otherwise we want to see if there is something available not too far
    for size in size_history:
        if cur_size <= size <= cur_size*(1 + pad_threshold):
            logger.debug(f"Input size {cur_size} not existing. Using close size of {size}")
            return size
    #If nothing exists we will have to recompile. We still use some padding to try limiting further recompilations if the size is reduced
    new_size = int(cur_size*(1 + pad_threshold/2) + 0.5)
    size_history.append(new_size)
    size_history.sort()
    logger.debug(f"Input size {cur_size} not existing. Creating new size of {new_size}")
    return new_size

@partial(jit, static_argnames=['fields'])
def simulate_drift(params, tracks, fields, rngkey):
    #Quenching and drifting
    new_tracks = quench(params, tracks, 2, fields)
    new_tracks = drift(params, new_tracks, fields)

    #Simulating the electron generation according to the diffusion coefficients
    electrons = generate_electrons(new_tracks, fields, rngkey)
    #Getting the pixels where the electrons are
    pIDs = get_pixels(params, electrons, fields)

    return electrons, pIDs

@jit
def get_renumbering(pIDs, unique_pixels):
    #Getting the renumbering of the pixels
    pIDs = pIDs.ravel()
    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs, method='sort')

    #Only getting the electrons for which the pixels are in the active region
    mask = (pix_renumbering < unique_pixels.size) & (unique_pixels[pix_renumbering] == pIDs)
    return mask, pix_renumbering

@partial(jit, static_argnames=['fields'])
def simulate_signals(params, electrons, mask_indices, pix_renumbering, unique_pixels, response, fields):
    pix_renumbering = jnp.take(pix_renumbering, mask_indices, mode='fill', fill_value=0)
    npix = (2*params.number_pix_neighbors + 1)**2
    elec_ids = mask_indices//npix
    electrons_renumbered = jnp.take(electrons, elec_ids, mode='fill', fill_value=0, axis=0)
    #Getting the pixel coordinates
    xpitch, ypitch, plane, eid = id2pixel(params, unique_pixels)
    pixels_coord = get_pixel_coordinates(params, xpitch, ypitch, plane)
    #Getting the right indices for the currents
    t0, currents_idx = current_lut(params, response, electrons_renumbered, pixels_coord[pix_renumbering], fields)
    npixels = unique_pixels.shape[0]
    nticks_wf = int(params.time_interval[1]/params.t_sampling) + 1 #Adding one first element to serve as a garbage collector
    wfs = jnp.zeros((npixels, nticks_wf))

    start_ticks = (t0/params.t_sampling + 0.5).astype(int) - params.time_window
    

    wfs = accumulate_signals(wfs, currents_idx, electrons_renumbered[:, fields.index("n_electrons")], response, pix_renumbering, start_ticks, params.signal_length)
    integral, ticks = get_adc_values(params, wfs[:, 1:]*params.e_charge)

    adcs = digitize(params, integral)
    return adcs, unique_pixels, ticks

@partial(jit, static_argnames=['fields'])
def simulate_signals_parametrized(params, electrons, pIDs, unique_pixels, fields):
    xpitch, ypitch, plane, eid = id2pixel(params, pIDs)
    
    pixels_coord = get_pixel_coordinates(params, xpitch, ypitch, plane)
    t0, signals = current_mc(params, electrons, pixels_coord, fields)

    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs)

    nticks_wf = int(params.time_interval[1]/params.t_sampling) + 1 #Adding one first element to serve as a garbage collector
    wfs = jnp.zeros((unique_pixels.shape[0], nticks_wf))

    
    start_ticks = (t0/params.t_sampling + 0.5).astype(int) - signals.shape[1]

    wfs = accumulate_signals_parametrized(wfs, signals, pix_renumbering, start_ticks)
    # return signals, pix_renumbering, wfs, start_ticks, unique_pixels
    integral, ticks = get_adc_values(params, wfs[:, 1:])
    adcs = digitize(params, integral)
    # return wfs, unique_pixels
    return adcs, unique_pixels, ticks

def simulate_parametrized(params, tracks, fields, rngkey = 0):
    electrons, pIDs = simulate_drift(params, tracks, fields, rngkey)
    pIDs = pIDs.ravel()
    unique_pixels = jnp.unique(pIDs)
    padded_size = pad_size(unique_pixels.shape[0], "unique_pixels")

    unique_pixels = jnp.sort(jnp.pad(unique_pixels, (0, padded_size - unique_pixels.shape[0]), mode='constant', constant_values=-1))
 

    return simulate_signals_parametrized(params, electrons, pIDs, unique_pixels, fields)
    


def simulate(params, response, tracks, fields, rngkey = 0):
    electrons, pIDs = simulate_drift(params, tracks, fields, rngkey)

    main_pixels = pIDs[:, 2*params.number_pix_neighbors*(params.number_pix_neighbors+1)] #Getting the main pixel
    #Sorting the pixels and getting the unique ones
    unique_pixels = jnp.unique(main_pixels.ravel())
    padded_size = pad_size(unique_pixels.shape[0], "unique_pixels")

    unique_pixels = jnp.sort(jnp.pad(unique_pixels, (0, padded_size - unique_pixels.shape[0]), mode='constant', constant_values=-1))

    mask, pix_renumbering = get_renumbering(pIDs, unique_pixels)
    mask_indices = jnp.nonzero(mask)[0]

    padded_size = pad_size(mask_indices.shape[0], "pix_renumbering")
    mask_indices = jnp.pad(mask_indices, (0, padded_size - mask_indices.shape[0]), mode='constant', constant_values=-1)

    # errors = checkify.user_checks | checkify.index_checks | checkify.float_checks
    # checked_f = checkify.checkify(accumulate_signals, errors=errors)
    # err, wfs = checked_f(wfs, currents_idx, electrons[:, fields.index("n_electrons")], response, pix_renumbering, start_ticks - earliest_tick, params.signal_length)
    # err.throw()

    return simulate_signals(params, electrons, mask_indices, pix_renumbering, unique_pixels, response, fields)
    
    # return wfs, unique_pixels
    # return adcs, unique_pixels, ticks, wfs[:, 1:], t0, currents_idx, electrons, pix_renumbering, start_ticks

def params_loss(params, response, ref, pixels_ref, ticks_ref, tracks, fields, rngkey=0, loss_fn=loss, **loss_kwargs):
    adcs, pixels, ticks = simulate(params, response, tracks, fields, rngkey)
    loss_val, aux = loss_fn(adcs, pixels, ticks, ref, pixels_ref, ticks_ref, fields, **loss_kwargs)
    return loss_val, aux

def params_loss_parametrized(params, ref, pixels_ref, ticks_ref, tracks, fields, rngkey=0, loss_fn=loss, **loss_kwargs):
    adcs, pixels, ticks = simulate_parametrized(params, tracks, fields, rngkey)
    loss_val, aux = loss_fn(adcs, pixels, ticks, ref, pixels_ref, ticks_ref, fields, **loss_kwargs)
    return loss_val, aux

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
    print(params.electron_sampling_resolution)
    tracks = chop_tracks(tracks, fields, params.electron_sampling_resolution)
    # tracks = tracks[:300000]
    tracks = jnp.array(tracks)

    # tracks = order_tracks_by_z(tracks, fields)
    # tracks = filter_tracks(tracks, fields)

    return tracks, fields, original_tracks