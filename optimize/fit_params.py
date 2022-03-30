import os, sys
larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .utils import get_id_map, all_sim, embed_adc_list
from .ranges import ranges
from larndsim.sim_with_grad import sim_with_grad
import torch

from tqdm import tqdm

class ParamFitter:
    def __init__(self, relevant_params, track_fields, track_chunk, pixel_chunk,
                 detector_props, pixel_layouts, load_checkpoint = None,
                 lr=None, optimizer=None, loss_fn=None):

        # If you have access to a GPU, sim works trivially/is much faster
        if torch.cuda.is_available():
            self.device = 'cuda'
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = 'cpu'
        self.track_fields = track_fields
        if type(relevant_params) == dict:
            self.relevant_params_list = list(relevant_params.keys())
            self.relevant_params_dict = relevant_params
        elif type(relevant_params) == list:
            self.relevant_params_list = relevant_params
            self.relevant_params_dict = None
        else:
            raise TypeError("relevant_params must be list of param names or dict with learning rates")

        is_continue = False
        if load_checkpoint is not None:
            history = pickle.load(open(load_checkpoint, "rb"))
            is_continue = True

        # Simulation object for target
        self.sim_target = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk)
        self.sim_target.load_detector_properties(detector_props, pixel_layouts)

        # Simulation object for iteration -- this is where gradient updates will happen
        self.sim_iter = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk)
        self.sim_iter.load_detector_properties(detector_props, pixel_layouts)

        # Normalize parameters to init at 1, or set to checkpointed values
        for param in self.relevant_params_list:
            if is_continue:
                setattr(self.sim_iter, param, history[param][-1])
            else:
                setattr(self.sim_iter, param, getattr(self.sim_iter, param)/ranges[param]['nom'])

        # Keep track of gradients in sim_iter
        self.sim_iter.track_gradients(self.relevant_params_list)

        # Placeholder simulation -- parameters will be set by un-normalizing sim_iter
        self.sim_physics = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk)
        self.sim_physics.load_detector_properties(detector_props, pixel_layouts)

        # Set up optimizer -- can pass in directly, or construct as SGD from relevant params and/or lr
        if optimizer is None:
            if self.relevant_params_dict is None:
                if lr is None:
                    raise ValueError("Need to specify lr for params")
                else:
                    self.optimizer = torch.optim.SGD([getattr(self.sim_iter, param) for param in self.relevant_params_list], lr=lr)
            else:
                 self.optimizer = torch.optim.SGD(self.relevant_params_dict)

        else:
            self.optimizer = optimizer

        # Set up loss function -- can pass in directly, or MSE by default
        if loss_fn is None:
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        if is_continue:
            self.training_history = history
        else:
            self.training_history = {}
            for param in self.relevant_params_list:
                self.training_history[param] = []
                self.training_history[param + '_target'] = []
                self.training_history[param + '_lr'] = [lr]
            self.training_history['losses'] = []


    def make_target_sim(self, seed=2):
        np.random.seed(seed)
        print("Constructing target param simulation")
        for param in self.relevant_params_list:
            param_val = np.random.uniform(low=ranges[param]['down'], 
                                          high=ranges[param]['up'])

            print(f'{param}, target: {param_val}, init {getattr(self.sim_target, param)}')    
            setattr(self.sim_target, param, param_val)

    def fit(self, dataloader, epochs=300, save_freq=1, print_freq=1):
        # make a folder for the pixel target
        if os.path.exists('target'):
            shutil.rmtree('target', ignore_errors=True)
        os.makedirs('target')


        # Include initial value in training history (if haven't loaded a checkpoint)
        for param in self.relevant_params_list:
            self.training_history[param + '_target'].append(getattr(self.sim_iter, param).item())
            #if len(self.training_history[param]) == 0:
            #    self.training_history[param].append(getattr(self.sim_iter, param).item())

        # The training loop
        with tqdm(total=len(dataloader) * epochs) as pbar:
            for epoch in range(epochs):

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
                        
                        target, pix_target = all_sim(self.sim_target, selected_tracks_torch, self.track_fields,
                                                    event_id_map, unique_eventIDs,
                                                    return_unique_pix=True)
                        embed_target = embed_adc_list(self.sim_target, target, pix_target)

                        torch.save(embed_target, 'target/batch' + str(i) + '_target.pt')

                    else:
                        embed_target = torch.load('target/batch' + str(i) + '_target.pt')

                    # Undo normalization (sim -> sim_physics)
                    for param in self.relevant_params_list:
                        setattr(self.sim_physics, param, getattr(self.sim_iter, param)*ranges[param]['nom'])

                    # Simulate and get output
                    output, pix_out = all_sim(self.sim_physics, selected_tracks_torch, self.track_fields,
                                              event_id_map, unique_eventIDs,
                                              return_unique_pix=True)

                    # Embed both output and target into "full" image space
                    embed_output = embed_adc_list(self.sim_physics, output, pix_out)

                    # Calc loss between simulated and target + backprop
                    loss = self.loss_fn(embed_output, embed_target)
                    loss.backward()

                    # To be investigated -- sometimes we get nans. Avoid doing a step if so
                    nan_check = torch.tensor([getattr(self.sim_iter, param).grad.isnan() for param in self.relevant_params_list]).sum()
                    if nan_check == 0 and not loss.isnan():
                        self.optimizer.step()
                        losses_batch.append(loss.item())

                    pbar.update(1)

                # Print out params at each epoch
                if epoch % print_freq == 0:
                    for param in self.relevant_params_list:
                        print(param, getattr(self.sim_physics,param).item())

                # Keep track of training history
                for param in self.relevant_params_list:
                    self.training_history[param].append(getattr(self.sim_iter, param).item())
                if len(losses_batch) > 0:
                    self.training_history['losses'].append(np.mean(losses_batch))

                # Save history in pkl files
                n_steps = len(self.training_history[param])
                if n_steps % save_freq == 0:
                    with open(f'history_epoch{n_steps}.pkl', "wb") as f_history:
                        pickle.dump(self.training_history, f_history)
                    if os.path.exists(f'history_epoch{n_steps-save_freq}.pkl'):
                        os.remove(f'history_epoch{n_steps-save_freq}.pkl')
