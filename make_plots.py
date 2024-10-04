#!/usr/bin/env python3

import argparse
import pickle
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

import numpy as np
from glob import glob
import os, sys
from optimize.ranges import ranges
import scipy
from scipy.ndimage import uniform_filter1d


# ----------------- Constants ----------------------- #
# set matplotlib color rotation to have more options, hexcolor website: https://www.colorhexa.com/
#            greyblue   orange    green    midred   grey prpl   BROWN     PINK     drkGREY   
hexcolors = ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'ffc0cb', '696969', 
             '00008b', 'ffaa88', '66ffaa', '8b0000', '17becf', '800080', 'bbbbbb',  '87ceeb', 'bcbd22', 'eeee00', '000000']
#             drk blue | peach  |  lime  |  drkred  |  cyan |   purple  | ltgrey | SKY BLUE | gry yellow | yellow | black
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])

# used when making dEdx histograms
h5_dict = '/sdf/home/b/bkroul/l-sim/h5/'
files_dict = {'muon':     h5_dict + 'old_data_cut.h5',
              'proton':   h5_dict +'first_data_cut_high_dEdx.h5',
              'dEdx > 5': h5_dict + 'proton_min-dEdx5.h5',
              'dEdx < 2': h5_dict + 'proton_max-dEdx2.h5',
              'proton_no_nuclei': h5_dict + 'proton_dEdx_no_nuclei.h5',
              'muon_raw': '/fs/ddn/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5',
              'proton_raw': '/fs/ddn/sdf/group/neutrino/cyifan/muon-sim/larndsim_output/f1_1000_p_high_KE/edepsim-output.h5'
             }

# labels for y-axis when plotting parameter iterations
labels = {'Ab' : "$A_{B}$",
          'kb' : "$k_{B}$ [g/cm$^2$/MeV]",
          'lifetime' : "$\\tau$ [$\mu s$]",
          'long_diff' : "$D_{L}$ [$cm^2/\mu s$]",
          'tran_diff' : "$D_{T}$ [$cm^2/\mu s$]",
          #'vdrift' : 'vdrift [$cm/\mu s$]', # we use link_vdrift_efield now
          'eField' : "$\\epsilon$ [kV/cm]"}

# dictionary of all particles found in input data so far
pdgIds = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6:"t", 7:"b'", 8:"t'", # quarks
          -11:"e+", 11: "e-", 12: "νe", -13: "µ+", 13: "µ-", 14: "νµ",  # leptons
          15: "τ-", 16: "ντ", 17: "τ'-", 18: "ντ'", 
          21: "g", 22: "γ", 23: "Z0", 24: "W+", 37: "H+", 39: "G", # bosons, graviton
          111: "π0", 211: "π+", -211: "π-", # light I = 0 mesons
          321: "K+", # strange mesons
          2112: "n", 2212: "p", 3112: "Σ−", 3122: "Λ", 3222: "Σ+", # light baryons, strange baryons
          1000010020: "deuteron", 1000010030: "triton", 1000010040: "H4", 1000020030: "He3", 1000020040: "He4", 1000020060: "He6",
          1000030040: 'Li4', 1000030060: 'Li6', 1000030070: 'Li7', 1000040080: 'Be8', 1000040090: 'Be9', 1000040100: 'Be10', 1000050080: 'B8', 1000050100: 'B10', 1000050110: 'B11', 1000060110: 'C11', 1000060120: 'C12', 1000060130: 'C13', 1000060140: 'C14', 1000070120: 'N12', 1000070140: 'N14', 1000070150: 'N15', 1000080160: 'O16', 1000080170: 'O17', 1000080180: 'O18', 1000080190: 'O19', 1000080200: 'O20', 1000090170: 'F17', 1000090180: 'F18', 1000090190: 'F19', 1000090200: 'F20', 1000100200: 'Ne20', 1000100210: 'Ne21', 1000100220: 'Ne22', 1000100230: 'Ne23', 1000100240: 'Ne24', 
          1000110220: 'Na22', 1000110230: 'Na23', 1000110240: 'Na24', 1000110241: 'Na24', 1000110250: 'Na25', 1000120230: 'Mg23', 1000120240: 'Mg24', 1000120250: 'Mg25', 1000120260: 'Mg26', 1000120270: 'Mg27', 1000120280: 'Mg28', 1000130260: 'Al26', 1000130270: 'Al27', 1000130280: 'Al28', 1000130290: 'Al29', 1000130300: 'Al30', 1000130310: 'Al31', 1000140270: 'Si27', 1000140280: 'Si28', 1000140290: 'Si29', 1000140300: 'Si30', 1000140310: 'Si31', 1000140320: 'Si32', 1000140330: 'Si33', 1000150300: 'P30', 1000150310: 'P31', 1000150320: 'P32', 1000150330: 'P33', 1000150340: 'P34', 1000150350: 'P35', 1000150360: 'P36', 1000160320: 'S32', 1000160330: 'S33', 1000160340: 'S34', 1000160350: 'S35', 1000160360: 'S36', 1000160370: 'S37', 1000160380: 'S38', 1000170340: 'Cl34', 1000170350: 'Cl35', 1000170360: 'Cl36', 1000170370: 'Cl37', 1000170380: 'Cl38', 1000170389: 'Cl38', 1000170390: 'Cl39', 1000170400: 'Cl40', 1000180350: 'Ar35', 1000180360: 'Ar36', 1000180370: 'Ar37', 1000180380: 'Ar38', 1000180390: 'Ar39', 1000180400: "Ar40", 1000180410: 'Ar41', 1000190380: 'K38', 1000190390: 'K39', 1000190400: 'K40', 1000190410: 'K41'
          }
#atoms[proton number]
atoms = "0,H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr".split(',')

# ----------------------- Utility Functions ---------------------- #
def nucleus_to_label(pdgId):
    """
    Returns "Ar40"-esque labels of atom + num baryons for nuclei
    Labeling scheme from https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
        10LZZZAAAI  -- pdgId nuclear code
        L = num lamba baryons
        ZZZ = num protons = atomic number
        AAA = total num baryons = protons + neutrons + lambda baryons
        I = isomer number = excitation level, 0 for ground state
    """
    pdgId = str(pdgId)
    if len(pdgId) != 10 or pdgId[0] != '1': 
        print(f"pdgId {pdgId} is not a nucleus!")
        return "N/A"
     # str(int( removes leading zeroes.
    L = pdgId[2]; Z = int(pdgId[3:6]); A = str(int(pdgId[6:9])); I = pdgId[9]
    return atoms[Z]+A

def tolerant_mean(arrays):
    """
    Returns a the moving average, max, and min of a numpy array of arrays 
    """
    lens = [arr.size for arr in arrays]
    all_arr = np.ma.empty( (np.max(lens),len(arrays)) )
    all_arr.mask = True
    for idx, arr in enumerate(arrays):
        all_arr[:lens[idx], idx] = arr
    return all_arr.mean(axis = -1), all_arr.max(axis = -1), all_arr.min(axis = -1)

def smooth(array, length): 
    """
    Smooths array over a length window just like scipy.uniform1d(array, length) while ignoring np.inf() values
    """
    l = len(array)
    new_array = np.zeros(l)
    for i in range(l):
        mi = max(i-length//2, 0)  # min valid array
        ma = min(i+length//2, l)  # max valid array
        sub_array = array[mi:ma]
        if mi == 0: # reflect at start # 0 1 2 3 4 5
            sub_array = np.append(array[0:length//2 - i], sub_array)
        if ma == l: # reflect at end
            sub_array = np.append(sub_array, array[2*l - 1 - i - length//2:l])         
        new_array[i] = np.mean(sub_array[sub_array != np.inf])  # ignore np.inf values
    return new_array

def movingavg(array):
    l = len(array)
    new_array = np.zeros(l)
    for i in range(l):
        sub_array = np.asarray(array[:i])
        sub_array = sub_array[sub_array != np.inf]  # ignore np.inf values
        sub_array = sub_array[sub_array != np.nan]
        new_array[i] = np.sum(sub_array) / (i+1)
    return new_array

# ----------------- Plotting Functions ----------------- #
def plot_losses(data, plot_name, unif_len, cut_to_min=False, print_info=False, label=None):
    """
    Plot the simulation loss, saved in data['loss']
    """
    if cut_to_min: # graph convergence on minimum iterations over all data
        min_iterations = min([len(dat['loss']) for dat in data])
    plt.figure(figsize=PLOT_FIGSIZE)
    for count, dat in enumerate(data):
        seed_init = dat['seed_init']; seed = dat['seed']; data_seed = dat['data_seed']
        loss = dat['loss'][:min_iterations] if cut_to_min else dat["loss"]
        l = f"seed {seed}" if label == 'seed' else (f"iseed {seed_init}" if label == "seed_init" else (f"dtseed {data_seed}" if data_seed else f"{seed}-{seed_init}-{data_seed}"))
        if print_info: print(f"\tloss {l}: {loss[0]}-->{loss[-1]}")
        plt.plot(loss, c=f'C{count}', linewidth=LINEWIDTH, label=l)
        #plt.plot(len(loss)+40, np.mean(loss[-unif_len:]), marker='_', linewidth=LINEWIDTH, markersize=6)
    l = f" Initial Seed {seed_init} " if label == 'seed' else (f" Seed {seed} " if label == "seed_init" else (f" Data Seed {data_seed} " if label == "data_seed" else ""))
    plt.title(f"Loss{l}")
    plt.ylabel('Simulation Loss')
    plt.xlabel('Training Iteration')
    plt.legend(loc='best', fontsize="10")
    plt.tight_layout()
    plt.savefig(plot_name, dpi=PLOT_DPI)
    print(f'Saving plot to {plot_name}')
    plt.close()

def plot_params(data, plot_name, param, label=None):
    """
    Plot parameter iterations for data, stored in dat['data']['{paaram}_iter']
    """
    targets = []; label_target = False
    plt.figure(figsize=PLOT_FIGSIZE)
    for count, dat in enumerate(data):
        # PLOT TARGET VALUE
        target_val = dat['data'][f'{param}_target'][0]
        if target_val not in targets: 
            targets.append(target_val)
            plt.plot([0, len(dat['data'][f"{param}_iter"])], [target_val]*2, c=f'C{count}', ls='dashed', label=("target: %.3e" % target_val if label_target else None), linewidth=LINEWIDTH*1.2)
        # plot & label iteration
        seed_init = dat['seed_init']; seed = dat['seed']; init_val = dat['data'][f"{param}_iter"][0]; data_seed = dat['data_seed']
        l = f"seed {seed}" if label == 'seed' else ("initial val %.3e" % init_val if label == "seed_init" else (f"dtseed {data_seed}" if label == 'data_seed' else f"{seed}-{seed_init}-{data_seed}"))
        plt.plot(dat['data'][f"{param}_iter"], c=f'C{count}', linewidth=LINEWIDTH, label=l)
        plt.plot(0, init_val, c=f'C{count}', marker='_', linewidth=LINEWIDTH, markersize=12)
    l = f"Initial Seed {seed_init} " if label == 'seed' else (f" Seed {seed} " if label == "seed_init" else (f" Data Seed {data_seed} " if label == "data_seed" else ""))
    plt.title(f"{param} {l}Iterations")
    plt.ylabel(f'Fitting {labels[param]}')
    plt.xlabel('Training Iteration')
    #plt.legend(loc='best', fontsize="10") # w/ legend ofc
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_name, dpi=PLOT_DPI)
    print(f'Saving plot to {plot_name}')
    plt.close()

def plot_convergences(data, plot_name, plot_individual_convergences=True, cut_to_min=False, print_info=False, logy=True, label=None,
                      iter_range=None):
    """
    Plot parameter convergence across all parameters in a run, stored in data['convergence']
     - plot_name = name of the plot to be saved as plot_name.{config.ext}}
     - plot_individual_convergences
       = True will plot the convergence of each individual run along with average, min, & max
       = False will only plot the average, min, & max convergence for the data
     - cut_to_min
       = True will only show convergences up to the minimum num of iterations in data
       = False will show convergences and calculate average as moving average over all iterations in data
     - print_info determines if initial -> final convergence info is printed for each run in data
     - logy determines if plot has a logarithmic y axis or not
     - label = 'seed' or 'seed_init' determines label description, if either data has same seed_init or same seed
     - iter_range determines iteration (x) range to plot
    """
    if cut_to_min: # graph convergence on minimum iterations over all data
        min_iterations = min([len(dat['convergence']) for dat in data])
    plt.figure(figsize=PLOT_FIGSIZE)
    if plot_individual_convergences:
        for count, dat in enumerate(data):
            seed = dat['seed']; seed_init = dat['seed_init']
            conv = dat['convergence']
            if print_info: print(f"convergence {seed}-{seed_init}: {conv[0]}-->{conv[-1]}")
            l = f"seed {seed}" if label == 'seed' else (f"initial seed {seed_init}" if label == 'seed_init' else (f"data seed {data_seed}" if label == 'data_seed' else f"{seed}-{seed_init}-{data_seed}"))
            if cut_to_min:
                plt.plot(conv[:min_iterations], c=f'C{count}', linewidth=LINEWIDTH, label=l)
            else:
                plt.plot(conv, c=f'C{count}', linewidth=LINEWIDTH, label=l)
    if cut_to_min:
        avg_iter = np.mean(data['convergence'], 0)
        max_iter = np.max(data['convergence'], 0)
        min_iter = np.min(data['convergence'], 0)
    else: # dynamically change min, max, avg
        avg_iter, max_iter, min_iter = tolerant_mean(data['convergence'])
    if not plot_individual_convergences:
        plt.plot(min_iter, color='#6e77ff', linewidth=LINEWIDTH*0.5)
        plt.plot(max_iter, color='#6e77ff', linewidth=LINEWIDTH*0.5)
        plt.plot(avg_iter, color='red', linewidth=LINEWIDTH*2)
    else:
        plt.plot(avg_iter, label='avg', color='red', linewidth=LINEWIDTH*2, ls='dashed')
        plt.legend(loc='best', fontsize='8')
    plt.fill_between(range(len(avg_iter)), min_iter, max_iter, alpha=0.1, color='#6e77ff')
    if logy: plt.yscale('log')
    l = f" Initial Seed {data['seed_init'][0]}" if label == 'seed' else (f" Seed {data['seed'][0]}" if label == "seed_init" else (f" Data Seed {data['data_seed'][0]}" if label == "data_seed" else ""))
    plt.title(f"Parameter Convergence{l}")
    plt.ylabel(f'Convergence to Target Parameters [%]')
    plt.xlabel('Training Iteration')
    if iter_range is not None:
        plt.xlim(iter_range)
    #plt.ylim(np.min(min_iter), np.max(max_iter))  # fit y max, min to max, min of data
    plt.tight_layout()
    plt.savefig(plot_name, dpi=PLOT_DPI)
    print(f'Saving plot to {plot_name}')
    plt.close()

def plot_elements(fname, print_info=False, xlog=True, ylog=True, nbins = 30, e_range=None):
    # return tracks masked by element
    elem_msk = lambda arr, e: [int(str(p)[3:6]) == e for p in arr['pdgId']]
    
    # get file, tracks
    if print_info: print(f'getting {fname}, {files_dict[fname]}')
    hfile = h5py.File(files_dict[fname])
    tracks = hfile['segments']
    # cut tracks to energy range & only nuclei
    if e_range == None:
        e_range = [1e-10,1e10] if xlog else [0,100]
    if print_info: print(tracks.size,"total tracks")
    tracks = tracks[np.logical_and(tracks['dEdx'] < e_range[1], tracks['dEdx'] > e_range[0])]
    tracks = tracks[tracks['pdgId'] > 1e6]
    
    # get particles & sort by descending number of counts
    nuclei, counts = np.unique(tracks['pdgId'], return_counts=True)
    nuclei = nuclei.tolist() # dont include '0' particle
    if len(nuclei) == 0:
        sys.exit("No nuclei found in "+fname+"!")
    nuclei.sort(reverse=True, key=lambda n: tracks[tracks['pdgId'] == n].size)
    elements = np.unique([int(str(p)[3:6]) for p in tracks['pdgId']]).tolist()
    elements.sort(reverse=True, key=lambda e: tracks[elem_msk(tracks, e)].size)
    if print_info: 
        print(tracks.size,"nuclei tracks")
        print(f"nuclei found in {fname}: "+", ".join(f"{pdgIds[nuclei[i]]}-{counts[i]}" for i in range(len(nuclei))))
    print(f"elements in {fname}: "+", ".join(atoms[e] for e in elements))
    # plot!
    plt.figure(figsize=PLOT_FIGSIZE)
    min_e = np.min(tracks['dEdx'])
    max_e = np.max(tracks['dEdx'])
    if xlog: # set plot xlog, ylog
        plt.xscale('log')
        bins = np.logspace(np.log10(min_e), np.log10(max_e), nbins)
    else:
        bins = np.linspace(min_e, max_e, nbins)
    if ylog: # want to see all particles
        plt.yscale('log')
    else:  
        elements = elements[::-1]
    
    for c, element in enumerate(elements):
        if print_info: print(f"plotting {element_name} {count}")
        element_name = atoms[element]
        energies = tracks[elem_msk(tracks, element)]['dEdx']
        count = energies.size
        if not ylog: c = len(elements) - 1 - c
        #weight = [1/ncounts[c]]*counts[c] if normalize else None
        plt.hist(energies, bins=bins, color=f"C{c}", alpha=1, label=f"{element_name}: {count}", stacked=True)
    plot_name = f"plot_dEdx_{fname}_elements_{('T' if xlog else 'F')}{('T' if ylog else 'F')}_{nbins}.png"
    plt.xlabel("dE/dx [MeV/cm]")
    plt.ylabel("# of entries")
    plt.title(f"{fname} elements dEdx, MeV/cm")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_name, dpi=PLOT_DPI)
    print("Saved plot to",plot_name,"\n")
    plt.close()

# plot particles, stacked, for a file. plot one for nucleus, one for not
def plot_particle_dEdxs(fname, xlog=True, ylog=True, nbins = 30, e_range=None, nuclei=None, print_info=False):
    # get file, tracks
    if print_info: print(f'getting {fname}, {files_dict[fname]}')
    hfile = h5py.File(files_dict[fname])
    tracks = hfile['segments']
    # cut tracks to energy range
    if e_range == None:
        e_range = [1e-10,1e10] if xlog else [0,100]
    tracks = tracks[np.logical_and(tracks['dEdx'] < e_range[1], tracks['dEdx'] > e_range[0])]
    # cut tracks to include or exclude nuclei
    add = "all"
    if nuclei: # plot only nuclei
        tracks = tracks[tracks['pdgId'] > 1e6]
        add = "nuclei"
    elif nuclei == False: # plot only non-nuclei
        tracks = tracks[tracks['pdgId'] < 1e6]
        add = "no-nuclei"
    # get particles & sort by descending number of counts
    particles = np.subtract(np.unique(tracks['pdgId']), np.array([0])).tolist() # dont include '0' particle
    particles.sort(reverse=True, key=lambda p: tracks[tracks['pdgId'] == p].size)
    print(f"particles found in {fname}: "+", ".join(pdgIds[p] for p in particles))

    # begin plotting
    min_e = np.min(tracks['dEdx'])
    max_e = np.max(tracks['dEdx'])
    plt.figure(figsize=PLOT_FIGSIZE)
    if xlog: # set plot xlog, ylog
        plt.xscale('log')
        bins = np.logspace(np.log10(min_e), np.log10(max_e), nbins)
    else:
        bins = np.linspace(min_e, max_e, nbins)
    if ylog: # want to see all particles
        plt.yscale('log')
    else:  
        particles = particles[::-1]
    # plot all particles
    for c, p in enumerate(particles):
        #print(f"plotting {pdgIds[p]}")
        if not ylog: c = len(particles) - c - 1
        energies = tracks[tracks['pdgId'] == p]['dEdx']
        count = energies.size
        #weight = [1/ncounts[c]]*counts[c] if normalize else None
        plt.hist(energies, bins=bins, color=f'C{c}', alpha=1, label=f"{pdgIds[p]}: {count}", stacked=True)
    fadd = ''.join([''.join([j[0] for j in i.split(' ')]) for i in fname.split('_')])
    plot_name = f"plot_dEdx_{fadd}_particles-{add}_{('T' if xlog else 'F')}{('T' if ylog else 'F')}_{nbins}.png"
    plt.xlabel("dE/dx [MeV/cm]")
    plt.ylabel("# of entries")
    plt.title(f"{fname} {add} particles dEdx, MeV/cm")
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(plot_name, dpi=PLOT_DPI)
    print("Saved plot to",plot_name,"\n")
    plt.close()

# Plot dEdxs comparing one or more edepsim-style h5 files
# if plot_particles, will make plots comparing each file 
#                        for each particle in both files
# normalize = normalize histogram counts relative to size of each set
def plot_dEdxs(fnames, plot_particles=False, print_info=False, normalize=True,
               xlog=False, ylog=True, nbins = 30, e_range=None):
    label_max_bin = False
    numfiles = len(fnames)
    hfiles = np.empty((numfiles,),dtype='object')
    tracks = np.empty((numfiles,),dtype='object')
    energies = np.empty((numfiles,),dtype='object')
    ncounts = np.empty((numfiles,),dtype='object')
    if e_range is None:
        e_range = [1e-4,1e10] if xlog else [0,100]
    
    for c, f in enumerate(fnames):
        hfiles[c] = h5py.File(files_dict[f])
        tracks[c] = hfiles[c]['segments']
        e_msk = np.logical_and(tracks[c]['dEdx'] < e_range[1],tracks[c]['dEdx'] > e_range[0])
        tracks[c] = tracks[c][e_msk]
    # get list of all particles among all files
    fadd = '.'.join([''.join([''.join([j[0] for j in i.split(' ')]) for i in f.split('_')]) for f in fnames])
    plot_name = f"plot_dEdx_{fadd}_{('T' if xlog else 'F')}{('T' if ylog else 'F')}_{nbins}{('N' if normalize else '')}.png"
    
    cmsk = [c for c in range(numfiles)]
    uniq_particle_list = ['nah']
    if plot_particles:
        cmsk = [c for c in range(numfiles) if 'pdgId' in tracks[c].dtype.names]
        print(cmsk)
        if len(cmsk):
            p = np.concatenate([tracks[c]['pdgId'] for c in cmsk])
            uniq_particle_list = np.subtract(np.unique(p), np.array([0]))
            # print new particles found so I can add them to pdgId list 
            pp = False 
            for p in uniq_particle_list:
                if p not in pdgIds.keys():
                    pp = True
                    pdgIds[p] = nucleus_to_label(p)
            if pp: print(pdgIds)
            print("particles found: "+", ".join(pdgIds[p] for p in uniq_particle_list))
        else:
            print('no files',' ,'.join(fnames),'have pdgIds.\n quitting now...')
            sys.exit()
    
    tracks_ = tracks.copy()
    for c in cmsk: ncounts[c] = tracks[c]['dEdx'].size # counts to normalize datasets
    counts = ncounts.copy()
    for particle in uniq_particle_list:
        # # # SET TRACKS (CUTTING FOR PARTICLE TRACKS) # # # #
        cmsk2 = [c for c in cmsk]
        if plot_particles: # CUT TO ONLY TRACKS WITH PARTICLE & within energy range
            cmsk2 = []
            for c in cmsk:
                msk = tracks[c]['pdgId'] == particle
                tracks_[c] = tracks[c][msk]
                if print_info: print(f"resized {tracks[c].size}-->{tracks_[c].size}")
                if tracks_[c].size > 0:
                    cmsk2.append(c)
            if len(cmsk2) < 2: continue
            plot_name = f"plot_dEdx_{fadd}_{pdgIds[particle]}_{('T' if xlog else 'F')}{('T' if ylog else 'F')}_{nbins}{('N' if normalize else '')}'.png"
        # # # SET ENERGIES AND COUNTS # # #
        for c in cmsk2:
            energies[c] = tracks_[c]['dEdx']
            counts[c] = energies[c].size
        add = pdgIds[particle]+" " if plot_particles else ""
        if print_info:
            string = ', '.join([ fnames[c]+"-"+str(np.sum(tracks_[c]['dx'])) for c in cmsk2])
            print(f"all {add}segments length: {string}")
            string = ', '.join([ fnames[c]+"-"+str(np.sum(energies[c])) for c in cmsk2])
            print(f"total {add}dEdx: {string}")
            string = ', '.join([ fnames[c]+"-"+str(counts[c]) for c in cmsk2])
            print(f"number of {add}segments: {string}")
        all_e = np.concatenate([energies[c] for c in cmsk2])
        #print(len(all_e), all_e.shape)
        min_e = np.min(all_e)
        max_e = np.max(all_e)
        #print(f"energy range: {min_e}<-->{max_e}")
        plt.figure(figsize=PLOT_FIGSIZE)
        if xlog: # set plot xlog, ylog
            plt.xscale('log')
            logbins = np.logspace(np.log10(min_e),np.log10(max_e),nbins)
        else:
            logbins = np.linspace(min_e,max_e,nbins)
        if ylog: plt.yscale('log')

        # PLOT NORMALIZED HISTOGRAM OF ENERGIES FOR ALL FILES
        for c in cmsk2: 
            weight = [1/ncounts[c]]*counts[c] if normalize else None
            (n, bins, patches) = plt.hist(np.asarray(energies[c]), bins=logbins, weights=weight, 
                                          color=f'C{c}', alpha=0.5, label=f"{fnames[c]}: {counts[c]}")
            if label_max_bin:
                max_counts_idx = max([i for i in range(len(n))], key = lambda x: n[x])
                patch = patches[max_counts_idx]
                label = "%.2e" % n[max_counts_idx] if normalize else str(n[max_counts_idx]).split('.')[0]
                plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height()+0.01, label,
                    ha='center', va='bottom')  # ADD LABEL TO MAX BIN IN FILE
        
        plt.xlabel("dE/dx [MeV/cm]")
        add = "normalized " if normalize else ""
        plt.ylabel(f"{add}# of entries")
        string = ' vs. '.join([fnames[c] for c in cmsk2])
        msk = pdgIds[particle]+" " if plot_particles else ""
        plt.title( f"{string} {msk}dEdx counts in MeV/cm" )
        plt.legend(loc='best')
        plt.savefig(plot_name, dpi=PLOT_DPI)
        print("Saved plot to",plot_name,"\n")
        plt.close()

# —.——.———.———.———.———.———.———.———.———.———.———.———.———.———.———.———.———.—————————.———.——————————————————————————————————————————
# ——.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.—.——————————————————————————————————————————
# ———.———.———.———.———.———.———.———.———.———.———.———.———.———.———.———.———.———.—.—.—.———.———————————————————————————————————————————
def main(config):
    config.plot = [t.lower() for t in config.plot]
    global LINEWIDTH, PLOT_DPI, PLOT_FIGSIZE
    LINEWIDTH    = config.linewidth if config.linewidth else .8
    PLOT_DPI     = 600
    PLOT_FIGSIZE = (6.4, 4.8) # default matplotlib figure size
    # google slides is PLOT_FIGSIZE = (10, 5.625)
    # significant number of seeds having one seed_init 
    #   to plot for a single seed_init, & vice versa for seeds 
    sig_num_to_plot = 3
    cut_to_min      = False  # cut graphs to minimum value?
    print_info      = True   # print more info while plotting?
    # GET UNIF_LEN FOR LOSSES SMOOTHING
    UNIF_LEN = int(config.plot[config.plot.index('loss')+1]) if config.plot[config.plot.index('loss')+1].isdigit() else 320
    # smooth with moving average of loss after smoothing with uniform?
    # (avg[loss[:i] for i in loss])
    do_moving_avg = 'avg' in config.plot[config.plot.index('loss')+1:min(config.plot.index('loss')+3,len(config.plot))]
   
    if "dedx" in config.plot:
        PLOT_FIGSIZE = (8,4.8)
        # number of histogram bins specified after dedx or default 200
        NUM = int(config.plot[config.plot.index('dedx')+1]) if config.plot[config.plot.index('dedx')+1].isdigit() else 200
       
        for f in ['dEdx > 5', 'dEdx < 2']:
            for log_combo in [(True, True), (False, False), (False, True)]: #(xlog, ylog)
                #plot_elements(f, xlog=log_combo[0], ylog=log_combo[1], nbins = NUM)
                plot_particle_dEdxs(f, xlog=log_combo[0], ylog=log_combo[1], nbins = NUM)
                #plot_dEdxs(['dEdx > 5', 'dEdx < 2'], xlog=log_combo[0], ylog=log_combo[1], nbins=NUM, normalize=True, 
                #           e_range=[1e-4,1e10], plot_particles=False)
        sys.exit() # dont look at any pickle data
     

    # -------------- LOAD DATA FROM .pkl FILES --------------------------------
    data_entry = np.dtype([('seed', 'i4'),('seed_init', 'i4'),('data_seed', 'i4'),('data', 'O'),('convergence', 'O'),('loss', 'O')])
    data = np.array([], dtype=data_entry)

    fnames = []
    for seed in config.seeds:
        # search for all relevant seeds
        for start in ['history', 'losses']:
            if seed == -1: # no seeds specified
                # get all relevant files regardless of seed
                fnames.extend(glob(f"{start}*{config.label}*.pkl"))
            else: 
                # try various combinations of seed, config_label
                for slabel in ['_dtseed','_seed','i=seed','i=dtseed']:
                    fnames.extend(glob(f'{start}*{slabel}{seed}_*{config.label}*.pkl'))
                    fnames.extend(glob(f'{start}*{config.label}_*{slabel}{seed}*.pkl'))
    fnames = list(set(fnames)) # dont repeat any filenames
        
    for f in fnames: 
        history = pickle.load(open(f, "rb"))
        # get seed, init_seed, and data_seed
        
        data_seed = 0
        if 'config' in history.keys():
            seed = history['config'].seed; seed_init = history['config'].seed_init; data_seed = history['config'].data_seed
        elif 'seed' in history.keys():
            seed = history['seed']; seed_init = history['seed_init']; data_seed = history['data_seed']
        else: # get seeds from filename lol
            if 'i=dt_seed' in f:
                seed = seed_init = f.split('i=dt_seed')[1][0]
            if 'i=seed' in f:
                seed = seed_init = f.split('i=seed')[1][0]
                if 'dtseed' in f:
                    data_seed = f.split('dtseed')[1][0]
            if 'iseed' in f:
                seed_init = f.split('iseed')[1][0]
        # get params
        if len(config.params) == 0: # if no --params label, automatically use all params available
            config.params = np.unique([key.split('_iter')[0] for key in history.keys() if key.split('_iter')[0] in labels])
            print('Params found:',', '.join(config.params))
        
        # STORE SMOOTHED LOSS IN dat['loss']
        loss = np.array(history['losses_iter'])
        if data_seed == 6: print(loss)
        if UNIF_LEN > 0:  loss = smooth(loss, UNIF_LEN) # smooth loss!
        if do_moving_avg: loss = movingavg(loss)

        # STORE PARAMETER CONVERGENCE dat['convergence']
        try: # if history[f"{param}_iter"] is stored as a single value
            float(history[f"{config.params[0]}_iter"])
            size = int(loss.size)
            norm_iters = np.zeros((len(config.params), size))
            for c, param in enumerate(config.params):
                norm_iters[c] = [abs(history[f"{param}_iter"] - history[f"{param}_target"])/history[f"{param}_target"]]*size
        except TypeError:
            min_iterations = min([len(history[f"{param}_iter"]) for param in config.params])
            norm_iters = np.zeros((len(config.params), min_iterations))
            for c, param in enumerate(config.params):
                norm_iters[c] = (abs(np.array(history[f"{param}_iter"]) - history[f'{param}_target'][0])/history[f'{param}_target'][0])[:min_iterations]
            
        if config.convergence == 'max':
            convergence = np.max(norm_iters, 0)
        elif config.convergence == 'sum':
            convergence = np.sum(norm_iters, 0)
        elif config.convergence == 'min':
            convergence = np.min(norm_iters, 0)
        elif config.convergence == 'mean':
            convergence = np.mean(norm_iters, 0)
        
        entry = np.array([(seed, seed_init, data_seed, history, convergence, loss)], dtype=data_entry)
        data = np.append(data, entry)
   
    if data.shape == (0,):
        sys.exit(f"Error: found no .pkl files in {os.getcwd()} with label '{config.label}' and seeds {config.seeds}")
    print("Data found: "+', '.join([f"{dat['seed']}-{dat['seed_init']}"+(f"-{dat['data_seed']}" if data_seed else "") for dat in data]))
    
    seed_inits = np.unique(data['seed_init'])
    seeds = np.unique(data['seed'])
    data_seeds = np.unique(data['data_seed'])
    msk = [len(data[data['seed'] == seed]) >= sig_num_to_plot for seed in seeds]
    msk1 = [len(data[data['seed_init'] == seed_init]) >= sig_num_to_plot for seed_init in seed_inits]
    msk2 = [len(data[data['data_seed'] == data_seed]) >= sig_num_to_plot for data_seed in data_seeds]
    plot_seeds = seeds[msk]
    plot_inits = seed_inits[msk1]
    plot_dtseeds = data_seeds[np.logical_and(msk2, [d != 0 for d in data_seeds])]
    # now use label for plotting purposes
    config.label = config.label + f"_seeds{'-'.join([str(s) for s in seeds])}"+config.label_add
    # multiple runs for each seed (varying initial condition), plot 
    # plot each param, showing iterations of all seeds ran

    # -------------------------- CALL PLOTTING SCRIPTS ---------------------------------------
    # plot simulation losses
    if 'loss' in config.plot:
        add = "-avg" if do_moving_avg else ""
        PLOT_FIGSIZE = (8,4.8)
        # PLOT INITIAL VALUE w/ MULT SEEDS
        for seed_init in plot_inits: 
            plot_losses(data[data['seed_init'] == seed_init], 
                        f"plot_loss{add}_iseed{seed_init}_{UNIF_LEN}_{config.label}.{config.ext}", UNIF_LEN, 
                        print_info = print_info, label='seed', cut_to_min=cut_to_min)
        # PLOT MULT INITAL VALUES per SEED
        for seed in plot_seeds: 
            plot_losses(data[data['seed'] == seed], 
                        f"plot_loss{add}_seed{seed}_{UNIF_LEN}_{config.label}.{config.ext}", UNIF_LEN, 
                        print_info = print_info, label='seed_init', cut_to_min=cut_to_min)
        for data_seed in plot_dtseeds: 
            plot_losses(data[data['data_seed'] == data_seed], 
                        f"plot_loss{add}_dtseed{data_seed}_{UNIF_LEN}_{config.label}.{config.ext}", UNIF_LEN, 
                        print_info = print_info, label='', cut_to_min=cut_to_min)
        # PLOT ALL LOSSES
        if len(plot_inits) + len(plot_seeds) + len(plot_dtseeds) != 1:
            plot_losses(data, f"plot_loss{add}_{UNIF_LEN}_{config.label}.{config.ext}", UNIF_LEN, 
                        print_info = print_info, cut_to_min=cut_to_min)

    # plot parameter iterations
    if 'all' in config.plot or "param" in config.plot:
        for param in config.params: 
            # VARYING INITIAL VALUE CONVERGING TO ONE TARGET, per SEED
            for seed in plot_seeds:
                plot_name = f'plot_vary-init_{param}_seed{seed}_{config.label}.{config.ext}'
                plot_params(data[data['seed'] == seed], plot_name, param, 'seed_init')
            # MULTIPLE TARGETS / SEEDS, ONE PLOT per initial seed
            for seed_init in plot_inits:  # only works for repeat !!!!!
                plot_name = f'plot_vary-seed_{param}_iseed{seed_init}_{config.label}.{config.ext}'
                plot_params(data[data['seed_init'] == seed_init], plot_name, param, 'seed')
            
            if len(plot_inits) + len(plot_seeds) != 1:
                plot_name = f'plot_{param}_{config.label}.{config.ext}'
                plot_params(data, plot_name, param)

    # plot parameter convergences
    if "convergence" in config.plot or "conv" in config.plot:
        logy = False
        #iter_range=[4500,5000] # determines range of x values to plot
        iter_range=None
        for logy in [False, True]:
            for plot_individual_convergences in [False, True]: 
                add = "-all" if plot_individual_convergences else ""
                add += "-regy" if not logy else ""

                for seed_init in plot_inits:
                    plot_name = f"plot_{config.convergence}{add}_iseed{seed_init}_{config.label}.{config.ext}"
                    plot_convergences(data[data['seed_init'] == seed_init], plot_name, plot_individual_convergences, cut_to_min, print_info, logy, 'seed', iter_range)
                
                for seed in plot_seeds:
                    plot_name = f"plot_{add}{config.convergence}_seed{seed}_{config.label}.{config.ext}"
                    plot_convergences(data[data['seed'] == seed], plot_name, plot_individual_convergences, cut_to_min, print_info, logy, 'seed_init', iter_range)
                for data_seed in plot_dtseeds: 
                    plot_name = f"plot_{add}{config.convergence}_dtseed{data_seed}_{config.label}.{config.ext}"
                    plot_convergences(data[data['data_seed'] == data_seed], plot_name, plot_individual_convergences, cut_to_min, print_info, logy, 'data_seed', iter_range)
                # plot all convergences if we havent plotted anything yet or have plotted more than one graph
                if len(plot_inits) + len(plot_seeds) + len(plot_dtseeds) != 1: 
                    plot_name = f"plot_{add}{config.convergence}_all_{config.label}.{config.ext}"
                    plot_convergences(data, plot_name, plot_individual_convergences, cut_to_min, print_info, logy, iter_range)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="params", default=[], nargs="+",
                        help="List of parameters to plot.")
    parser.add_argument("--label", dest="label", default="",
                        help="Label of pkl file (after seed part).") 
    parser.add_argument("--seeds", dest="seeds", default=[-1], nargs="+",
                        help="List of target seeds to plot.") 
    parser.add_argument("--ext", dest="ext", default="png",
                        help="Image extension (e.g., pdf or png)") 
    parser.add_argument("--convergence", dest='convergence', default='max',
                        help="How normalized parameter iterations are combined into a total convergence level: 'sum', 'max', 'min', 'mean'")
    parser.add_argument("--plot", dest='plot', default=[], nargs="+",
                        help="List of plot specifications: \nloss [UNIF_LEN] [avg]: make loss plots, [int length for smoothing] [make moving average loss plot] \nall: plot parameter iterations, \
                              \nconv: make parameter convergence plots\n dedx [NUM_BINS]: make histogram of energy data [int number of histogram bins], edit configurations in main() 471 \
                              ")
    parser.add_argument("--linewidth", dest='linewidth', default=None,
                        help="List of plot specifications: \nloss: make loss plots, \nall: plot parameter iterations, \nconvergence: make parameter convergence plots")
    parser.add_argument("--ladd", dest='label_add', default='',
                        help="List of plot specifications: \nloss: make loss plots, \nall: plot parameter iterations, \nconvergence: make parameter convergence plots")
    args = parser.parse_args()
    main(args)