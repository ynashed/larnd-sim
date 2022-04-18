import os, sys
import shutil
import pickle
import numpy as np
from utils import get_id_map
from ranges import ranges
from sim_module import SimModule
import torch
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm

class DistDataParallelWrapper(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class ParamFitter:
    def __init__(self, relevant_params, track_fields, track_chunk, pixel_chunk,
                 detector_props, pixel_layouts, job_id,
                 local_rank=0, world_rank=0, world_size=1,
                 load_checkpoint = None, lr=None, optimizer=None, loss_fn=None):

        # If you have access to a GPU, sim works trivially/is much faster
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = 'cpu'
        self.job_id = job_id
        self.world_rank = world_rank
        self.track_fields = track_fields
        if type(relevant_params) == dict:
            self.relevant_params_list = list(relevant_params.keys())
            self.relevant_params_dict = relevant_params
        elif type(relevant_params) == list:
            self.relevant_params_list = relevant_params
            self.relevant_params_dict = None
        else:
            raise TypeError("relevant_params must be list of param names or dict with learning rates")

        history = None
        if load_checkpoint is not None:
            history = pickle.load(open(load_checkpoint, "rb"))

        # Simulation object for target
        self.sim_target = SimModule(track_chunk=track_chunk, pixel_chunk=pixel_chunk,
                                    detector_props=detector_props, pixel_layouts=pixel_layouts)

        # Simulation object for iteration -- this is where gradient updates will happen
        self.sim_iter = SimModule(track_chunk=track_chunk, pixel_chunk=pixel_chunk,
                                  detector_props=detector_props, pixel_layouts=pixel_layouts)
        # Normalize parameters to init at 1, or set to checkpointed values
        self.sim_iter.init_params(self.relevant_params_list, history)
        self.sim_iter.track_params(self.relevant_params_list)
        self.sim_iter.to(self.device)
        if world_size > 1:
            self.sim_iter = DistDataParallelWrapper(self.sim_iter,
                                                    find_unused_parameters=True,
                                                    device_ids=[local_rank],
                                                    output_device=[local_rank])

        # Set up optimizer -- can pass in directly, or construct as SGD from relevant params and/or lr
        if optimizer is None:
            if self.relevant_params_dict is None:
                if lr is None:
                    raise ValueError("Need to specify lr for params")
                else:
                    self.optimizer = torch.optim.SGD(self.sim_iter.get_params(self.relevant_params_list), lr=lr)
            else:
                 self.optimizer = torch.optim.SGD(self.relevant_params_dict)

        else:
            self.optimizer = optimizer

        # Set up loss function -- can pass in directly, or MSE by default
        if loss_fn is None:
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        if history is not None:
            self.training_history = history
        else:
            self.training_history = {}
            for param in self.relevant_params_list:
                self.training_history[param] = []
            self.training_history['losses'] = []


    def make_target_sim(self, seed=2):
        np.random.seed(seed)
        print("Constructing target param simulation")
        for param in self.relevant_params_list:
            param_val = np.random.uniform(low=ranges[param]['down'], 
                                          high=ranges[param]['up'])

            print(f'{param}, target: {param_val}, init {self.sim_target.get_params([param])}')
            self.sim_target.set_param(param, param_val)

    def fit(self, dataloader, sampler, epochs=300, save_freq=5, print_freq=1):
        # make a folder for the pixel target
        if os.path.exists(f'target_{self.job_id}_{self.world_rank}'):
            shutil.rmtree(f'target_{self.job_id}_{self.world_rank}', ignore_errors=True)
        os.makedirs(f'target_{self.job_id}_{self.world_rank}')
        # Include initial value in training history (if haven't loaded a checkpoint)
        for param in self.relevant_params_list:
            if len(self.training_history[param]) == 0:
                self.training_history[param].append(self.sim_iter.get_params([param])[0].item())

        # The training loop
        with tqdm(total=len(dataloader) * epochs) as pbar:
            for epoch in range(epochs):
                if sampler is not None:
                    sampler.set_epoch(epoch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                # Losses for each batch -- used to compute epoch loss
                losses_batch=[]
                for i, selected_tracks_torch in enumerate(dataloader):
                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Get rid of the extra dimension and padding elements for the loaded data
                    selected_tracks_torch = torch.flatten(selected_tracks_torch, start_dim=0, end_dim=1)
                    selected_tracks_torch = selected_tracks_torch[selected_tracks_torch[:, self.track_fields.index("dx")] > 0]
                    event_id_map, unique_eventIDs = get_id_map(selected_tracks_torch, self.track_fields, self.device)
                    selected_tracks_torch = selected_tracks_torch.to(self.device)

                    # Simulate target and store them
                    if epoch == 0:
                        target, pix_target = self.sim_target(selected_tracks_torch, self.track_fields,
                                                             event_id_map, unique_eventIDs,
                                                             return_unique_pix=True)
                        embed_target = self.sim_target.embed_adc_list(target, pix_target)

                        torch.save(embed_target, f'target_{self.job_id}_{self.world_rank}/batch{i}_target.pt')

                    else:
                        embed_target = torch.load(f'target_{self.job_id}_{self.world_rank}/batch{i}_target.pt')


                    # Simulate and get output
                    output, pix_out = self.sim_iter(selected_tracks_torch, self.track_fields,
                                              event_id_map, unique_eventIDs,
                                              return_unique_pix=True)

                    # Embed both output and target into "full" image space
                    embed_output = self.sim_iter.embed_adc_list(output, pix_out)

                    # Calc loss between simulated and target + backprop
                    loss = self.loss_fn(embed_output, embed_target)
                    loss.backward()

                    # To be investigated -- sometimes we get nans. Avoid doing a step if so
                    nan_check = torch.tensor([self.sim_iter.get_params([param])[0].grad.isnan()
                                              for param in self.relevant_params_list]).sum()
                    if nan_check == 0 and loss !=0 and not loss.isnan():
                        self.optimizer.step()
                        losses_batch.append(loss.item())

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    pbar.update(1)

                # Print out params at each epoch
                if epoch % print_freq == 0:
                    for param in self.relevant_params_list:
                        print(param, self.sim_iter.get_params([param])[0].item())

                # Keep track of training history
                for param in self.relevant_params_list:
                    self.training_history[param].append(self.sim_iter.get_params([param])[0].item())
                if len(losses_batch) > 0:
                    self.training_history['losses'].append(np.mean(losses_batch))

                # Save history in pkl files
                n_steps = len(self.training_history[param])
                if n_steps % save_freq == 0:
                    with open(f'history_{self.job_id}_rank{self.world_rank}_epoch{n_steps}.pkl', "wb") as f_history:
                        pickle.dump(self.training_history, f_history)
