#!/usr/bin/env python
# coding: utf-8



# This is need so you can import larndsim without doing python setup.py install
from glob import glob
import os,sys,inspect
import shutil
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,'/sdf/group/neutrino/cyifan/convergence/larnd_sim_2')


import matplotlib.pyplot as plt
from matplotlib import cm, colors
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np
import eagerpy as ep
import h5py

import matplotlib as mpl
import pickle
import math

from numpy.lib import recfunctions as rfn
import torch

from tqdm import tqdm
import time

from larndsim.sim_with_grad import sim_with_grad


def torch_from_structured(tracks):
    tracks_np = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)
    return torch.from_numpy(tracks_np).float()

def structered_from_torch(tracks_torch, dtype):
    return rfn.unstructured_to_structured(tracks_torch.cpu().numpy(), dtype=dtype)


# If you have access to a GPU, sim works trivially/is much faster
if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'

    
# ### Dataset import
# First of all we load the `edep-sim` output. For this sample we need to invert $z$ and $y$ axes.

dir_name = '/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/'
# fname = dir_name + "edep_reco_larndsim_output_100_fakedata_allevts_dEdx1.6_unitlen10.h5"
fname = dir_name + "edepsim-output.h5"
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


# Let's say we only take tracks that are longer than roughly 5 cm 
# (we shall see if this causes problem for S1)

index = {}
all_events = np.unique(tracks['eventID'])
total_n_trks = 0
for ev in all_events:
    trk_pool = []
    track_set = np.unique(tracks[tracks['eventID'] == ev]['trackID'])
    for trk in track_set:
        trk_msk = (tracks['eventID'] == ev) & (tracks['trackID'] == trk)
#         if sum(tracks[trk_msk]['dx']) > 5:
        if max(tracks[trk_msk]['z']) - min(tracks[trk_msk]['z']) > 30:
            trk_pool.append(trk)
    index[ev] = np.array(trk_pool)
    total_n_trks += len(trk_pool)
    
print("number of qualified tracks in this sample: " + str(total_n_trks))


# Define number of tracks to train with
# <= total_n_trks
#n_tracks_to_train = total_n_trks

# utility random seed (torch for FEE noise and the target value)
rdn_seed_util = 6666

## Time to draw lotteries
#rdn_seed = 2
#np.random.seed(rdn_seed)
#train_index = []
#temp_index = index.copy()
#
#for i_trk in range(n_tracks_to_train):
#    rand_ev = np.random.choice(list(temp_index.keys()))
#    while temp_index[rand_ev].size == 0:
#        temp_index.pop(rand_ev, None)
#        rand_ev = np.random.choice(list(temp_index.keys()))
#    rand_track = np.random.choice(temp_index[rand_ev])
#    
#    temp_index[rand_ev] = np.delete(temp_index[rand_ev],np.where(temp_index[rand_ev] == rand_track)) 
#    train_index.append(np.array([rand_ev, rand_track])) 
#
#train_index = np.array(train_index)
#print(train_index)

train_index = np.array([[58, 4]])
print(train_index)
n_tracks_to_train = len(train_index)

def batch(tracks, train_index, i_batch, size=1):
    
    i_trk = 0
    batch_index = []
    while i_batch * size + i_trk < min(i_batch * size + size, n_tracks_to_train):
        print("i_batch: " + str(i_batch) + ", ev: " + str(train_index[i_batch * size + i_trk][0]) + ", trk: " + str(train_index[i_batch * size + i_trk][1]))
        if i_trk == 0:
            mask = ((tracks['eventID']== train_index[i_batch * size + i_trk][0]) & (tracks['trackID'] == train_index[i_batch * size + i_trk][1]))
        else:
            mask = ((tracks['eventID']== train_index[i_batch * size + i_trk][0]) & (tracks['trackID'] == train_index[i_batch * size + i_trk][1])) | mask
     
        batch_index.append(np.array([train_index[i_batch * size + i_trk]]))
        i_trk += 1
        
    mask = mask & (abs(tracks['z']) < 27.5)
    
    
    this_out_trk = tracks[mask]
    print("this_out_trk x: ", this_out_trk['x'])
    print("this_out_trk y: ", this_out_trk['y'])
    print("this_out_trk z: ", this_out_trk['z'])
    print("seg: ", len(this_out_trk))
    print("")
    #this_out_trk = this_out_trk[6:9]
    out_trk = torch_from_structured(this_out_trk.copy())
    
    return np.array(batch_index), out_trk 


# ## Simulation
# To flexibly keep track of parameters/gradients, simulations are housed in a class `sim_with_grad`. This is derived from class versions of all the other modules. Parameters are housed in `consts`, with method `track_gradients` to promote the constants to `requires_grad=True` PyTorch tensors.



# ## The simulation
# Following the flow of the simulation chain, define a function which takes in the `sim_with_grad` object, runs whatever pieces of the simulation, and returns desired output.

def get_id_map(selected_tracks, fields):
    # Here we build a map between tracks and event IDs (no param dependence, so np should be ok)
    unique_eventIDs = np.unique(selected_tracks[:, fields.index('eventID')])
    event_id_map = np.searchsorted(unique_eventIDs,np.asarray(selected_tracks[:, fields.index('eventID')]))
    
    event_id_map_torch = torch.from_numpy(event_id_map).to(device)
    return event_id_map_torch, unique_eventIDs


def all_sim(sim, selected_tracks, fields, event_id_map, unique_eventIDs, return_unique_pix=False):    
    selected_tracks_quench = sim.quench(selected_tracks, sim.birks, fields=fields)
    selected_tracks_drift = sim.drift(selected_tracks_quench, fields=fields)

    active_pixels_torch, neighboring_pixels_torch, n_pixels_list_ep = sim.get_pixels(selected_tracks_drift,
                                                                                     fields=fields)

    track_starts_torch, max_length_torch = sim.time_intervals(event_id_map, 
                                                              selected_tracks_drift, 
                                                              fields=fields)
    
    signals_ep = sim.tracks_current(neighboring_pixels_torch, selected_tracks_drift, 
                                          max_length_torch,
                                          fields=fields)

    unique_pix_torch = torch.empty((0, 2))
    pixels_signals_torch = torch.zeros((len(unique_pix_torch), len(sim.time_ticks)*50))

    shapes_torch = neighboring_pixels_torch.shape
    joined_torch = neighboring_pixels_torch.reshape(shapes_torch[0]*shapes_torch[1], 2)

    this_unique_pix_torch = torch.unique(joined_torch, dim=0)
    this_unique_pix_torch = this_unique_pix_torch[(this_unique_pix_torch[:,0] != -1) & (this_unique_pix_torch[:,1] != -1),:]
    unique_pix_torch = torch.cat((unique_pix_torch, this_unique_pix_torch),dim=0)

    this_pixels_signals_torch = torch.zeros((len(this_unique_pix_torch), len(sim.time_ticks)*50))
    pixels_signals_torch = torch.cat((pixels_signals_torch, this_pixels_signals_torch), dim=0)

    pixel_index_map_torch = torch.full((selected_tracks.shape[0], neighboring_pixels_torch.shape[1]), -1)
    compare_torch = (neighboring_pixels_torch[..., np.newaxis, :] == unique_pix_torch)

    indices_torch = torch.where(torch.logical_and(compare_torch[..., 0], compare_torch[...,1]))
    pixel_index_map_torch[indices_torch[0], indices_torch[1]] = indices_torch[2]
    
    pixels_signals_torch = sim.sum_pixel_signals(pixels_signals_torch,
                                                 signals_ep,
                                                track_starts_torch,
                                                pixel_index_map_torch)
    
    time_ticks_torch = torch.linspace(0, len(unique_eventIDs)*sim.time_interval[1]*3, pixels_signals_torch.shape[1]+1)

    integral_list_torch, adc_ticks_list_torch = sim.get_adc_values(pixels_signals_torch,
                                                                   time_ticks_torch,
                                                                   0)
    adc_list_torch = sim.digitize(integral_list_torch)

    if return_unique_pix:
        return adc_list_torch, unique_pix_torch,
    else:
        return adc_list_torch

# Update parameters for training loop
def update_grad_param(sim, name, value):
    setattr(sim, name, value)
    sim.track_gradients([name])

# ADC counts given as list of pixels. Better for loss to embed this in the "full" pixel space
def embed_adc_list(sim, adc_list, unique_pix):
    zero_val = sim.digitize(torch.tensor(0)).item()
    new_list = torch.ones((sim.n_pixels[0], sim.n_pixels[1], adc_list.shape[1]))*zero_val

    plane_id = unique_pix[..., 0] // sim.n_pixels[0]
    unique_pix[..., 0] = unique_pix[..., 0] - sim.n_pixels[0] * plane_id

    new_list[unique_pix[:, 0].long(), unique_pix[:, 1].long(), :] = adc_list
    
    return new_list


# ## Define dict with ranges from the spreadsheet
# https://docs.google.com/spreadsheets/d/1DLpSDgPsHeHUWCEBayYCcbLzIzd30vfBe72N-Z5vWTc/edit#gid=1247026028
ranges = {}
ranges['lArDensity']     = {'nom': 1.38, 'down': 1.37, 'up': 1.41}
ranges['eField']         = {'nom': 0.5, 'down': 0.45, 'up': 0.55}
ranges['vdrift']         = {'nom': 0.1587, 'down': 0.1400, 'up': 0.1800}
ranges['MeVToElectrons'] = {'nom': 4.237e4, 'down': 3.48e4, 'up': 5.13e4}
ranges['alpha']          = {'nom': 0.93, 'down': 0.85, 'up': 1.1}
ranges['beta']           = {'nom': 0.207, 'down': 0.18, 'up': 0.22}
ranges['Ab']             = {'nom': 0.8, 'down': 0.78, 'up': 0.88}
ranges['kb']             = {'nom': 0.0486, 'down': 0.04, 'up': 0.07}
ranges['lifetime']       = {'nom': 2.2e3, 'down': 300, 'up': 3e4}
ranges['long_diff']      = {'nom': 4.0e-6, 'down': 2e-6, 'up': 9e-6}
ranges['tran_diff']      = {'nom': 8.8e-6, 'down': 4e-6, 'up': 14e-6}


def param_l2_reg(param, sim):
    sigma = (ranges[param]['up'] - ranges[param]['down'])/2.
    return ((ranges[param]['nom']-getattr(sim, param))**2)/(sigma**2)


def calc_reg_loss(param_list, sim, regs):
    reg_loss = 0.
    for param in param_list:
        reg_loss+=regs[param]*param_l2_reg(param, sim)
        
    return reg_loss


#Simulate with some set:
sim_target = sim_with_grad(track_chunk=1, pixel_chunk=1)
sim_target.load_detector_properties("../larndsim/detector_properties/module0.yaml",
                             "../larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml")

relevant_params = ['lifetime' ]


#Setup simulation object for training -- params initialized to defaults
sim = sim_with_grad(track_chunk=1, pixel_chunk=1)
sim.load_detector_properties("../larndsim/detector_properties/module0.yaml",
                             "../larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml")

sim.track_gradients(relevant_params)


# ## Instead of varying all params by hand, draw randomly in range
# These are used as the targets

np.random.seed(rdn_seed_util)
for param in relevant_params:
    param_val = np.random.uniform(low=ranges[param]['down'], 
                                      high=ranges[param]['up'])
    
#     setattr(sim_target, param, param_val)
    setattr(sim_target, param, ranges[param]['nom'])

for param in relevant_params:
    print(f'{param}, target: {getattr(sim_target, param)}, init {getattr(sim, param).item()}')


# ### Add in rough checkpointing

do_checkpoint=False


if do_checkpoint:
    saved = glob('history_epoch*.pkl')
    num = max([int(os.path.splitext(file)[0][len('history_epoch'):]) for file in saved])
    history = pickle.load(open(f'history_epoch{num}.pkl', "rb"))

#Setup simulation object for training -- params initialized to defaults
sim = sim_with_grad(track_chunk=1, pixel_chunk=1)
sim.load_detector_properties("../larndsim/detector_properties/module0.yaml",
                             "../larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml")
if do_checkpoint:
    for param in relevant_params:
        setattr(sim, param, history[param][-1])


#Setup simulation object for training -- params initialized to defaults
sim_extra = sim_with_grad(track_chunk=1, pixel_chunk=1)
sim_extra.load_detector_properties("../larndsim/detector_properties/module0.yaml",
                             "../larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml")

regs = {}
regs['eField'] = 1e-4
regs['lifetime'] = 1e-4
regs['vdrift'] = 1e-4
regs['lArDensity'] = 1e-4
regs['Ab'] = 1e-4
regs['kb'] = 1e-4


if not do_checkpoint:
    for param in relevant_params:
        setattr(sim, param, getattr(sim, param)/ranges[param]['nom'])
    
sim.track_gradients(relevant_params)


#Simple MSE loss between target and output
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([#{'params' : sim.lArDensity, 'lr': 0.2},
#                               {'params' : sim.eField, 'lr': 8e0},
                              #{'params' : sim.vdrift, 'lr': 8e0},
                             # {'params' : sim.MeVToElectrons, 'lr': 0.001},
#                               {'params' : sim.Ab, 'lr': 8e0},
#                               {'params' : sim.kb, 'lr': 8e0},
                              {'params' : sim.lifetime, 'lr': 8e0},
#                              {'params' : sim.long_diff, 'lr': 1e-9},
#                               {'params' : sim.tran_diff, 'lr': 5e-9}
                              ])

                              

training_step_track = {}
for param in relevant_params:
    training_step_track[param] = []
# losses = []
# reg_losses = []

n_lin = 100

for param in relevant_params:
    if do_checkpoint:
        training_step_track[param] += history[param]
    else:
        training_step_track[param].append(getattr(sim, param).item())


    scan_set = np.linspace(ranges[param]['down'], ranges[param]['up'], n_lin)

    if os.path.exists(f'scan_target_{param}'):
        shutil.rmtree(f'scan_target_{param}', ignore_errors=True)
    os.makedirs(f'scan_target_{param}')
        
losses = []
batch_size = 1
# i_batch_in_track = 0
scan_start = time.time()
#for i_par, par in enumerate(scan_set):
for i_par, par in enumerate(np.array([ranges['lifetime']['nom']])):
    active_train_index = np.copy(train_index)
    print("----i_par: ", i_par)
    i_par_start = time.time()

    losses_iter=[]
    n_batch = math.ceil(n_tracks_to_train / batch_size)
    
    for i_batch in range(n_batch):
        t_batch_start = time.time()

        batch_index, selected_tracks_torch = batch(tracks, train_index, i_batch, batch_size)

        event_id_map, unique_eventIDs = get_id_map(selected_tracks_torch, tracks.dtype.names)
        selected_tracks_torch = selected_tracks_torch.to(device)

        if i_par == 0:
            torch.random.manual_seed(rdn_seed_util)
            target, pix_target = all_sim(sim_target, selected_tracks_torch, tracks.dtype.names, 
                                         event_id_map, unique_eventIDs,
                                      return_unique_pix=True)
            embed_target = embed_adc_list(sim_target, target, pix_target)

            torch.save(embed_target, f'scan_target_{relevant_params[0]}/batch' + str(i_batch) + '_target.pt')
        else:
            embed_target = torch.load(f'scan_target_{relevant_params[0]}/batch' + str(i_batch) + '_target.pt')
            

        for param in relevant_params:
            setattr(sim_extra, param, par)

        #Simulate with that parameter and get output
        torch.random.manual_seed(i_batch)
        output, pix_out = all_sim(sim_extra, selected_tracks_torch, tracks.dtype.names, 
                                  event_id_map, unique_eventIDs,
                                  return_unique_pix=True)

        embed_output = embed_adc_list(sim, output, pix_out)

        #Calc loss between simulated and target + backprop
        loss = loss_fn(embed_output, embed_target) #+ calc_reg_loss(relevant_params, sim, regs)

        embed_output_mask = torch.gt(embed_output, 74.3657)
        embed_target_mask = torch.gt(embed_target, 74.3657)
        diff_mask = torch.eq(embed_output, embed_target)
        print("embed_output: ", embed_output[embed_output_mask])
        print("embed_target: ", embed_target[embed_target_mask])
        print("diff embed_output: ", embed_output[diff_mask == False])
        print("diff embed_target: ", embed_target[diff_mask == False])

        if not loss.isnan():
            losses_iter.append(loss.item())
            print("i_batch: " + str(i_batch) + ", loss of this track: " + str(loss.item()))

        i_batch += 1

    i_par_end = time.time()
    print("time of this iteration: " + str(round(i_par_end - i_par_start, 2)) + " s")

    if len(losses_iter) > 0:
        losses.append(np.mean(losses_iter))
    else:
        losses.append(-99)
    print("--loss at this value: " + str(losses[-1]))

scan_end = time.time()
print("scan time: " + str(round(scan_end - scan_start, 2)) + " s")

#np.save(f'{relevant_params[0]}_losses_scan_S0_lin{n_lin}_{n_tracks_to_train}trk_rdn{rdn_seed}_tgtrdn{rdn_seed_util}', np.asarray(losses))









