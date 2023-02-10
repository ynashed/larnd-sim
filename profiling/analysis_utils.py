import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from glob import glob
from dataclasses import dataclass, field
import pprint
import re

@dataclass
class Period:
    start: int = 0
    stop: int = 0
    id: int = 0
    note: dict = field(default_factory=dict)

    @property
    def time(self) -> int:
        return self.note['time']

@dataclass
class Epoch(Period):
    batches: list = field(default_factory=list)

@dataclass
class Batch(Period):
    events: list = field(default_factory=list)
    backprop: dict = field(default_factory=dict)
    backprop_memory: float = 0
    duration: int = 0

@dataclass
class Event(Period):
    actions: list = field(default_factory=list)
    memory: float = 0
    peak_memory_record: dict = field(default_factory=dict)
    time: float = 0
    time_max: int = 0
    pixels_dim: tuple = field(default_factory=tuple)
    
    @property
    def shape(self) -> tuple:
        return tuple(self.note['shape'])

    @property
    def dxs(self) -> np.array:
        return self.note['dxs'].numpy()

    @property
    def sim(self) -> dict:
        return next((note for note in self.actions if note['event'] == 'sim'), dict())

    @property
    def loss(self) -> dict:
        return next((note for note in self.actions if note['event'] == 'loss'), dict())

    @property
    def x_end(self) -> np.array:
        return self.note['x_end'].numpy()
    
    @property
    def x_start(self) -> np.array:
        return self.note['x_start'].numpy()
    
    @property
    def y_end(self) -> np.array:
        return self.note['y_end'].numpy()

    @property
    def y_start(self) -> np.array:
        return self.note['y_start'].numpy()

    @property
    def z_end(self) -> np.array:
        return self.note['z_end'].numpy()

    @property
    def z_start(self) -> np.array:
        return self.note['z_start'].numpy()

    @property
    def x_diffs(self) -> np.array:
        return np.abs(self.x_start - self.x_end)

    @property
    def y_diffs(self) -> np.array:
        return np.abs(self.y_start - self.y_end)

    @property
    def z_diffs(self) -> np.array:
        return np.abs(self.z_start - self.z_end)

    @property
    def volumes(self) -> np.array:
        return self.x_diffs*self.y_diffs*self.z_diffs

class Analyzer:
    def __init__(self, path) -> None:
        self.epochs = []
        self.nb_records = 0
        self.code_infos = {}
        self._load_data(path)


    def _load_data(self, path) -> None:
        for code_infos, notes, line_records in tqdm(import_checkpoints(path)):
            self._process_notes(notes)
            self._fill_events(line_records)
            self.nb_records += len(line_records)
            self.code_infos = code_infos


    def _process_notes(self, notes: list) -> None:
        for note in notes:
            ev = note['event']
            # print(ev)
            if ev == 'new_epoch':
                epoch = Epoch()
                epoch.start = note['prev_record']
                epoch.note = note
                # Filling the stop of the previous record
                if self.epochs:
                    self.epochs[-1].stop = note['prev_record'] - 1
                self.epochs.append(epoch)
            elif ev == 'new_batch':
                batch = Batch()
                batch.start = note['prev_record']
                batch.id = note['batch']
                batch.note = note
                if self.epochs and self.epochs[-1].batches:
                    self.epochs[-1].batches[-1].stop = note['prev_record'] - 1
                    if self.epochs[-1].batches[-1].events:
                        self.epochs[-1].batches[-1].events[-1].stop = note['prev_record'] - 1
                self.epochs[-1].batches.append(batch)
            elif ev == 'new_event':
                event = Event()
                event.start = note['prev_record']
                event.id = note['ev']
                event.note = note
                if self.epochs and self.epochs[-1].batches and self.epochs[-1].batches[-1].events:
                    self.epochs[-1].batches[-1].events[-1].stop = note['prev_record'] - 1
                self.epochs[-1].batches[-1].events.append(event)
            elif ev == 'backprop':
                if self.epochs and self.epochs[-1].batches:
                    self.epochs[-1].batches[-1].backprop = note
            else:
                if self.epochs and self.epochs[-1].batches and self.epochs[-1].batches[-1].events:
                    self.epochs[-1].batches[-1].events[-1].actions.append(note)

        for i in range(len(self.epochs[-1].batches) - 1):
            self.epochs[-1].batches[i].duration = self.epochs[-1].batches[i+1].time - self.epochs[-1].batches[i].time

            # if self.epochs:
            #     self.epochs[-1].stop = len(line_records)
        return self.epochs


    def _fill_events(self, records: list) -> None:
        for epoch in self.epochs:
            for batch in epoch.batches:
                for event in batch.events:
                    if event.memory == 0 and event.time == 0: #Only want to process the new ones
                        event.memory, event.peak_memory_record = get_sim_memory(event, records, self.nb_records)
                        event.time = get_sim_time(event, records, self.nb_records)
                        track_current_note = next((note for note in reversed(event.actions) if note['event'] == 'track_current'), dict()) #Picking the last one as the first one could be for the first sim without grads but shouldn,t change anything in fact
                        if track_current_note:
                            event.time_max = track_current_note['time_max']
                            event.pixels_dim = tuple(track_current_note['pixels_dim'])
                if batch.backprop_memory == 0:
                    batch.backprop_memory = records.loc[batch.backprop['prev_record'] + 1 - self.nb_records]['active_bytes.all.peak']/1024/1024/1024

    def print_epochs(self) -> None:
        for epoch in self.epochs:
            print('Epoch', epoch.start, epoch.stop, len(epoch.batches))
            for batch in epoch.batches:
                print('  ', 'Batch', batch.start, batch.stop, len(batch.events))
                for event in batch.events:
                    print('    ', 'Event', event.start, event.stop, len(event.actions))
                    for action in event.actions:
                        print('      ', action['event'])

def get_all_events(epochs: list[Epoch]) -> list[Event]:
    all_events = []
    for epoch in epochs:
        for batch in epoch.batches:
            all_events.extend(batch.events)
    return all_events

def get_sim_memory(event: Event, records: pd.DataFrame, first_record: int = 0) -> tuple[int, dict]:
    sim_note = event.sim
    loss_note = event.loss
    start_record = sim_note['prev_record'] + 1 - first_record
    stop_record = loss_note['prev_record'] - first_record

    idx = start_record + np.argmax(records.loc[start_record:stop_record]['active_bytes.all.peak'])
    return (records.loc[idx]['active_bytes.all.peak'], records.loc[idx])

def get_sim_time(event: Event, records: pd.DataFrame, first_record: int = 0) -> int:
    sim_note = event.sim
    loss_note = event.loss
    start_record = sim_note['prev_record'] + 1 -first_record
    stop_record = loss_note['prev_record'] -first_record

    return records.loc[stop_record]['time'] - records.loc[start_record]['time']
               


def sort_files(filelist: list[str]) -> list[str]:
    regex = r'_(\d+)\.'
    fileno = [int(re.search(regex, fname).group(1)) for fname in filelist]

    return [fname for _, fname in sorted(zip(fileno, filelist))]


def import_checkpoints(basename: str) -> tuple[list, list, pd.DataFrame]:
    if basename.endswith('.pkl'): #We got a single file
        files = [basename]
    else:
        files = glob(f"{basename}_*.pkl")
        files = sort_files(files)
    assert(len(files) != 0)
    for ifile in files:
        yield import_pkl(ifile)

def import_pkl(fname: str) -> tuple:
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return (obj['code_infos'], obj['notes'], pd.DataFrame(obj['line_records']))

def get_number_checkpoints(event: Event) -> tuple[int, int, int]:
    nb = 0
    max_ip = 0
    max_it = 0
    for action in event.actions:
        if action['event'] == 'checkpoint':
            nb += 1
            max_ip = max(max_ip, action['ip'])
            max_it = max(max_it, action['it'])
    return (nb, max_ip, max_it)

def evd(event: Event) -> None:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(event.dxs)):
        ax.plot([event.x_start[i], event.x_end[i]], [event.y_start[i], event.y_end[i]], zs=[event.z_start[i], event.z_end[i]])
    ax.set_xlim3d(-30, 30)
    ax.set_ylim3d(-80, 40)
    ax.set_zlim3d(-30, 30)
    ax.set_aspect('equal')

def get_code_line(record: dict, code_infos: dict) -> tuple[str, str]:
    code_hash = record['code_hash']
    line = record['line']

    func = code_infos[code_hash]['func']
    start_line = func.start_line

    return(func.__qualname__, func.lines[line-start_line])

def make_plots(epochs: list[Epoch]) -> None:
    plt.rcParams['font.size'] = 16

    events = get_all_events(epochs)
    sum_dxs = [np.sum(event.dxs) for event in events]
    max_dxs = [np.max(event.dxs) for event in events]
    memory = [event.memory/1024/1024 for event in events]
    time = [event.time/1e9 for event in events]
    nsegments = [len(event.dxs) for event in events]
    sum_volumes= [np.sum(event.volumes) for event in events]
    max_volumes= [np.max(event.volumes) for event in events]
    sum_diffx = [np.sum(event.x_diffs) for event in events]
    sum_diffy = [np.sum(event.y_diffs) for event in events]
    sum_diffz = [np.sum(event.z_diffs) for event in events]
    max_diffx = [np.max(event.x_diffs) for event in events]
    max_diffy = [np.max(event.y_diffs) for event in events]
    max_diffz = [np.max(event.z_diffs) for event in events]
    nb_checkpoints = [get_number_checkpoints(event)[0] for event in events]
    max_ip = [get_number_checkpoints(event)[1] for event in events]
    max_it = [get_number_checkpoints(event)[2] for event in events]
    max_cos = [np.max(np.array(event.z_diffs)/np.array(event.dxs)) for event in events]
    # time_max = [event.time_max.item() for event in events]
    time_max = [event.time_max for event in events]
    # pixel_dim = [np.prod(event.pixels_dim) for event in events]
    # pixel_dim = [event.pixels_dim[1] for event in events]
    pixel_dim = [0 for event in events]

    # print(events[0])

    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    plt.sca(axs[0])
    plt.scatter(sum_dxs, memory)
    plt.xlabel("Total dx of event [cm]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[1])
    plt.scatter(max_dxs, memory)
    plt.xlabel("Max dx of event [cm]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[2])
    plt.scatter(nsegments, memory)
    plt.xlabel("Number of segments")
    plt.ylabel("GPU memory [Mib]")
    fig.tight_layout()


    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    plt.sca(axs[0])
    plt.scatter(sum_dxs, time)
    plt.xlabel("Total dx of event [cm]")
    plt.ylabel("Time [s]")

    plt.sca(axs[1])
    plt.scatter(max_dxs, time)
    plt.xlabel("Max dx of event [cm]")
    plt.ylabel("Time [s]")

    plt.sca(axs[2])
    plt.scatter(nsegments, time)
    plt.xlabel("Number of segments")
    plt.ylabel("Time [s]")
    fig.tight_layout()

    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    plt.sca(axs[0])
    plt.scatter(sum_volumes, memory)
    plt.xlabel("Total volume of event [cm^3]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[1])
    plt.scatter(max_volumes, memory)
    plt.xlabel("Max volums of event [cm^3]")
    plt.ylabel("GPU memory [Mib]")

    # plt.sca(axs[2])
    # plt.scatter(nsegments, memory)
    # plt.xlabel("Number of segments")
    # plt.ylabel("GPU memory [Mib]")
    fig.tight_layout()

    #Sum diff
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    plt.sca(axs[0])
    plt.scatter(sum_diffx, memory)
    plt.xlabel("Total DeltaX of event [cm]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[1])
    plt.scatter(sum_diffy, memory)
    plt.xlabel("Total DeltaY of event [cm]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[2])
    plt.scatter(sum_diffz, memory)
    plt.xlabel("Total DeltaZ of event [cm]")
    plt.ylabel("GPU memory [Mib]")
    fig.tight_layout()


    #Max diff
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    
    plt.sca(axs[0])
    plt.scatter(max_diffx, memory)
    plt.xlabel("Max DeltaX of event [cm]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[1])
    plt.scatter(max_diffy, memory)
    plt.xlabel("Max DeltaY of event [cm]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[2])
    plt.scatter(max_diffz, memory)
    plt.xlabel("Max DeltaZ of event [cm]")
    plt.ylabel("GPU memory [Mib]")
    fig.tight_layout()


    ###############################################
    ####################TIME#######################
    ###############################################

    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    plt.sca(axs[0])
    plt.scatter(sum_volumes, time)
    plt.xlabel("Total volume of event [cm^3]")
    plt.ylabel("GPU time [s]")

    plt.sca(axs[1])
    plt.scatter(max_volumes, time)
    plt.xlabel("Max volums of event [cm^3]")
    plt.ylabel("GPU time [s]")

    plt.sca(axs[2])
    plt.scatter(max_cos, memory)
    plt.xlabel("Maximal cose of event")
    plt.ylabel("GPU memory [Mib]")
    plt.axvline(0.966, color='red')
    fig.tight_layout()

    #Sum diff
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    plt.sca(axs[0])
    plt.scatter(sum_diffx, time)
    plt.xlabel("Total DeltaX of event [cm]")
    plt.ylabel("GPU time [s]")

    plt.sca(axs[1])
    plt.scatter(sum_diffy, time)
    plt.xlabel("Total DeltaY of event [cm]")
    plt.ylabel("GPU time [s]")

    plt.sca(axs[2])
    plt.scatter(sum_diffz, time)
    plt.xlabel("Total DeltaZ of event [cm]")
    plt.ylabel("GPU time [s]")
    fig.tight_layout()


    #Max diff
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    
    plt.sca(axs[0])
    plt.scatter(max_diffx, time)
    plt.xlabel("Max DeltaX of event [cm]")
    plt.ylabel("GPU time [s]")

    plt.sca(axs[1])
    plt.scatter(max_diffy, time)
    plt.xlabel("Max DeltaY of event [cm]")
    plt.ylabel("GPU time [s]")

    plt.sca(axs[2])
    plt.scatter(max_diffz, time)
    plt.xlabel("Max DeltaZ of event [cm]")
    plt.ylabel("GPU time [s]")
    fig.tight_layout()

    fig = plt.figure()
    plt.scatter(time, memory)
    plt.xlabel("GPU time [s]")
    plt.ylabel("GPU memory [MiB]")
    fig.tight_layout()

    fig = plt.figure()
    plt.scatter(np.sqrt(np.array(max_diffy)**2 + np.array(max_diffx)**2), max_diffz, c=memory)
    plt.xlabel("Max ds [cm]")
    plt.ylabel("dz [cm]")
    plt.colorbar()
    fig.tight_layout()


    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    
    plt.sca(axs[0])
    plt.scatter(nb_checkpoints, memory)
    plt.xlabel("Nb checkpoints")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[1])
    plt.scatter(max_ip, memory)
    plt.xlabel("Max ip")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[2])
    plt.scatter(max_it, memory)
    plt.xlabel("Max it")
    plt.ylabel("GPU memory [Mib]")
    fig.tight_layout()


    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    
    plt.sca(axs[0])
    plt.scatter(time_max, memory)
    plt.xlabel("Time max [a.u.]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[1])
    plt.scatter(max_diffz, time_max, c=memory)
    plt.xlabel("Max DeltaZ of event [cm]")
    plt.ylabel("Time max [a.u.]")

    plt.sca(axs[2])
    plt.scatter(pixel_dim, memory)
    plt.xlabel("Dimension of pixel array")
    plt.ylabel("GPU memory [Mib]")
    fig.tight_layout()


    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    
    plt.sca(axs[0])
    plt.scatter(np.array(time_max)*np.array(nsegments), memory)
    plt.xlabel("Time max * Nsegments [a.u.]")
    plt.ylabel("GPU memory [Mib]")

    plt.sca(axs[1])
    plt.scatter(np.array(time_max)*np.array(nsegments)*np.array(pixel_dim), memory)
    plt.xlabel("Time max * Nsegments *Npixels [a.u.]")
    plt.ylabel("GPU memory [Mib]")

    # plt.sca(axs[2])
    # plt.scatter(pixel_dim, memory)
    # plt.xlabel("Dimension of pixel array")
    # plt.ylabel("GPU memory [Mib]")
    fig.tight_layout()

    plt.figure()
    plt.scatter(max_cos, memory)
    plt.xlabel(r"Maximal cos $\theta$ for segments of event")
    plt.ylabel("GPU memory [Mib]")
    plt.axvline(0.966, color='r', label="0.966")
    plt.legend()

    plt.tight_layout()

def plot_batches(epoch: Epoch) -> None:
    backprop_mem = [batch.backprop_memory for batch in epoch.batches]
    batch_duration = [batch.duration*1e-9 for batch in epoch.batches]

    #Memory
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(backprop_mem)), np.array(backprop_mem))
    plt.xlabel("Batch nb")
    plt.ylabel("GPU memory [Gio]")

    #Duration
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(batch_duration)), np.array(batch_duration))
    plt.xlabel("Batch nb")
    plt.ylabel("Duration [s]")

    #Duration cumulative
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(batch_duration)), np.cumsum(batch_duration))
    plt.xlabel("Batch nb")
    plt.ylabel("Duration [s]")
