import os, sys
larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .utils import get_id_map, all_sim, embed_adc_list, calc_loss, calc_soft_dtw_loss
from .ranges import ranges
from larndsim.sim_with_grad import sim_with_grad
import torch

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def normalize_param(param_val, param_name, scheme="divide", undo_norm=False):
    if scheme == "divide":
        if undo_norm:
            out_val = param_val * ranges[param_name]['nom']
        else:
            out_val = param_val / ranges[param_name]['nom']

        return out_val
    elif scheme == "standard":
        sigma = (ranges[param_name]['up'] - ranges[param_name]['down']) / 2.

        if undo_norm:
            out_val = param_val*sigma**2 + ranges[param_name]['nom']
        else:
            out_val = (param_val - ranges[param_name]['nom']) / sigma**2
            
        return out_val
    elif scheme == "none":
        return param_val
    else:
        raise ValueError(f"No normalization method called {scheme}")

class ParamFitter:
    def __init__(self, relevant_params, track_fields, track_chunk, pixel_chunk,
                 detector_props, pixel_layouts, load_checkpoint = None,
                 lr=None, optimizer=None, lr_scheduler=None, lr_kw=None, 
                 loss_fn=None, readout_noise_target=True, readout_noise_guess=False, 
                 out_label="", norm_scheme="divide", max_clip_norm_val=None, fit_diffs=False, optimizer_fn="Adam", 
                 no_adc=False, shift_no_fit=[], link_vdrift_eField=False, batch_memory=None, skip_pixels = False,
                 set_target_vals=[], vary_init=False, seed_init=30,
                 config = {}):

        if optimizer_fn == "Adam":
            self.optimizer_fn = torch.optim.Adam
        elif optimizer_fn == "SGD":
            self.optimizer_fn = torch.optim.SGD
        else:
            raise NotImplementedError("Only SGD and Adam supported")
        self.optimizer_fn_name = optimizer_fn

        self.no_adc = no_adc
        self.shift_no_fit = shift_no_fit
        self.link_vdrift_eField = link_vdrift_eField
        self.batch_memory = batch_memory
        self.skip_pixels = skip_pixels

        self.out_label = out_label
        self.norm_scheme = norm_scheme
        self.max_clip_norm_val = max_clip_norm_val
        if self.max_clip_norm_val is not None:
            logger.info(f"Will clip gradient norm at {self.max_clip_norm_val}")

        self.fit_diffs = fit_diffs
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
            raise TypeError("relevant_params must be list of param names or list of dicts with learning rates")

        is_continue = False
        if load_checkpoint is not None:
            history = pickle.load(open(load_checkpoint, "rb"))
            is_continue = True

        self.target_val_dict = None
        if len(set_target_vals) > 0:
            if len(set_target_vals) % 2 != 0:
                raise ValueError("Incorrect format for set_target_vals!")
            
            self.target_val_dict = {}
            for i_val in range(len(set_target_vals)//2):
                param_name = set_target_vals[2*i_val]
                param_val = set_target_vals[2*i_val+1]
                self.target_val_dict[param_name] = float(param_val)



        # Simulation object for target
        self.sim_target = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=readout_noise_target, skip_pixels=self.skip_pixels)
        self.sim_target.load_detector_properties(detector_props, pixel_layouts)
        self.sim_target.link_vdrift_eField = link_vdrift_eField

        # Simulation object for iteration -- this is where gradient updates will happen
        self.sim_iter = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=readout_noise_guess, skip_pixels=self.skip_pixels)
        self.sim_iter.load_detector_properties(detector_props, pixel_layouts)

        # Normalize parameters to init at 1, or random, or set to checkpointed values
        np.random.seed(seed_init)
        for param in self.relevant_params_list:
            if is_continue:
                setattr(self.sim_iter, param, normalize_param(history[param][-1], param, scheme=self.norm_scheme))
            else:
                if vary_init:
                    logger.info("Running with random initial guess")
                    init_val = np.random.uniform(low=ranges[param]['down'], 
                                                  high=ranges[param]['up'])
                    setattr(self.sim_iter, param, normalize_param(init_val, param, scheme=self.norm_scheme))
                else:
                    setattr(self.sim_iter, param, normalize_param(getattr(self.sim_iter, param), param, scheme=self.norm_scheme))
                
        # Placeholder simulation -- parameters will be set by un-normalizing sim_iter
        self.sim_physics = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=readout_noise_guess, skip_pixels=self.skip_pixels)
        self.sim_physics.load_detector_properties(detector_props, pixel_layouts)
        self.sim_physics.link_vdrift_eField = link_vdrift_eField

        for param in self.relevant_params_list:
            setattr(self.sim_physics, param, normalize_param(getattr(self.sim_iter, param), param, scheme=self.norm_scheme, undo_norm=True))

        # Keep track of gradients in sim_iter
        self.sim_iter.track_gradients(self.relevant_params_list, fit_diffs=self.fit_diffs)


        # Set up optimizer -- can pass in directly, or construct as SGD from relevant params and/or lr
        lr_dict = {}
        if optimizer is None:
            if self.relevant_params_dict is None:
                if lr is None:
                    raise ValueError("Need to specify lr for params")
                else:
                    if self.fit_diffs:
                        self.optimizer = self.optimizer_fn([getattr(self.sim_iter, param+"_diff") for param in self.relevant_params_list], lr=lr)
                    else:
                        self.optimizer = self.optimizer_fn([getattr(self.sim_iter, param) for param in self.relevant_params_list], lr=lr)
                    for param in self.relevant_params_list:
                        lr_dict[param] = lr
            else:
                param_config_list = []
                for param in self.relevant_params_dict.keys():
                    if self.fit_diffs:
                        param_config_list.append({'params': [getattr(self.sim_iter, param+"_diff")], 'lr' : float(self.relevant_params_dict[param])})
                    else:
                        param_config_list.append({'params': [getattr(self.sim_iter, param)], 'lr' : float(self.relevant_params_dict[param])})
                    lr_dict[param] = float(self.relevant_params_dict[param])
                self.optimizer = self.optimizer_fn(param_config_list)

        else:
            self.optimizer = optimizer

        if lr_scheduler is not None and lr_kw is not None:
            lr_scheduler_fn = getattr(torch.optim.lr_scheduler, lr_scheduler)
            self.lr_scheduler = lr_scheduler_fn(self.optimizer, **lr_kw)
            logger.info(f"Using learning rate scheduler {lr_scheduler}")
        else:
            self.lr_scheduler = None

        # Set up loss function -- can pass in directly, or choose a named one
        if loss_fn is None or loss_fn == "space_match":
            self.loss_fn = calc_loss
            self.loss_fn_kw = { 
                                'sim': self.sim_physics, 
                                'return_components' : False,
                                'no_adc' : self.no_adc 
                              }
            logger.info("Using space match loss")
        elif loss_fn == "SDTW":
            self.loss_fn = calc_soft_dtw_loss

            t_only = self.no_adc
            adc_only = not t_only
            # once cuda implementation in soft_dtw_cuda.py is set up
            #use_cuda = self.device == 'cuda'

            self.loss_fn_kw = {
            #                    'use_cuda' : use_cuda,
                                'adc_only' : adc_only,
                                't_only' : t_only,
                                'gamma' : 1
                              }
            if t_only:
                logger.info("Using Soft DTW loss on t only")
            else:
                logger.info("Using Soft DTW loss on ADC only")
        else:
            self.loss_fn = loss_fn
            self.loss_fn_kw = {}
            logger.info("Using custom loss function")

        if is_continue:
            self.training_history = history
        else:
            self.training_history = {}
            for param in self.relevant_params_list:
                self.training_history[param] = []
                self.training_history[param+"_grad"] = []
                self.training_history[param+"_iter"] = []

                self.training_history[param + '_target'] = []
                self.training_history[param + '_init'] = [normalize_param(getattr(self.sim_iter, param).item(), param, scheme=self.norm_scheme, undo_norm=True)]
                self.training_history[param + '_lr'] = [lr_dict[param]]
            for param in self.shift_no_fit:
                self.training_history[param + '_target'] = []

            self.training_history['losses'] = []
            self.training_history['losses_iter'] = []
            self.training_history['norm_scheme'] = self.norm_scheme
            self.training_history['fit_diffs'] = self.fit_diffs
            self.training_history['optimizer_fn_name'] = self.optimizer_fn_name

        self.training_history['config'] = config

    def make_target_sim(self, seed=2, fixed_range=None):
        np.random.seed(seed)
        logger.info("Constructing target param simulation")

        if self.target_val_dict is not None:
            if set(self.relevant_params_list + self.shift_no_fit) != set(self.target_val_dict.keys()):
                logger.debug(set(self.relevant_params_list + self.shift_no_fit))
                logger.debug(set(self.target_val_dict.keys()))
                raise ValueError("Must specify all parameters if explicitly setting target")

            logger.info("Explicitly setting targets:")
            for param in self.target_val_dict.keys():
                param_val = self.target_val_dict[param]
                logger.info(f'{param}, target: {param_val}, init {getattr(self.sim_physics, param)}')    
                setattr(self.sim_target, param, param_val)
        else:
            for param in self.relevant_params_list + self.shift_no_fit:
                if fixed_range is not None:
                    param_val = np.random.uniform(low=ranges[param]['nom']*(1.-fixed_range), 
                                                high=ranges[param]['nom']*(1.+fixed_range))
                else:
                    param_val = np.random.uniform(low=ranges[param]['down'], 
                                                high=ranges[param]['up'])

                logger.info(f'{param}, target: {param_val}, init {getattr(self.sim_physics, param)}')    
                setattr(self.sim_target, param, param_val)


    def optimize_batch_memory(self, sim, tracks) -> None:
        if self.batch_memory is not None:
            estimated_memory = sim.estimate_peak_memory(tracks, self.track_fields)
            chunk_size = int(self.batch_memory // estimated_memory)
            chunk_size = max(1, chunk_size) #Min value should be 1
            logger.info(f"Initial maximum memory of {estimated_memory/1024:.2f}Gio. Setting pixel_chunk_size to {chunk_size} and expect a maximum memory of {chunk_size*estimated_memory/1024:.2f}Gio")
            sim.update_chunk_sizes(1, chunk_size)

    
    def fit(self, dataloader, epochs=300, iterations=None, shuffle=False, 
            save_freq=10, print_freq=1):
        # If explicit number of iterations, scale epochs accordingly
        if iterations is not None:
            epochs = iterations // len(dataloader) + 1

        # make a folder for the pixel target
        if os.path.exists('target_' + self.out_label):
            shutil.rmtree('target_' + self.out_label, ignore_errors=True)
        os.makedirs('target_' + self.out_label)

        # make a folder for the fit result
        if not os.path.exists('fit_result'):
            os.makedirs('fit_result')

        # Include initial value in training history (if haven't loaded a checkpoint)
        for param in self.relevant_params_list:
            if len(self.training_history[param]) == 0:
                self.training_history[param].append(getattr(self.sim_physics, param))
                self.training_history[param+'_target'].append(getattr(self.sim_target, param))
            if len(self.training_history[param+"_iter"]) == 0:
                self.training_history[param+"_iter"].append(getattr(self.sim_physics, param))
        for param in self.shift_no_fit:
            if len(self.training_history[param+'_target']) == 0:
                self.training_history[param+'_target'].append(getattr(self.sim_target, param))

        if iterations is not None:
            pbar_total = iterations
        else:
            pbar_total = len(dataloader) * epochs

        # The training loop
        total_iter = 0
        with tqdm(total=pbar_total) as pbar:
            for epoch in range(epochs):
                # Losses for each batch -- used to compute epoch loss
                losses_batch=[]
                for i, selected_tracks_bt_torch in enumerate(dataloader):
                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Get rid of the extra dimension and padding elements for the loaded data
                    selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                    selected_tracks_bt_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("dx")] > 0]
                    event_id_map, unique_eventIDs = get_id_map(selected_tracks_bt_torch, self.track_fields, self.device)

                    loss_ev = []
                    # Calculate loss per event
                    for ev in unique_eventIDs:
                        selected_tracks_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("eventID")] == ev]

                        # flatten the dEdx and dE for the iteration input
                        selected_tracks_torch_target = selected_tracks_torch
                        selected_tracks_torch_output = selected_tracks_torch

                        selected_tracks_torch_output[:, self.track_fields.index('dEdx')] = 2
                        selected_tracks_torch_output[:, self.track_fields.index('dE')] = 2 * selected_tracks_torch_output[:, self.track_fields.index('dx')]

                        selected_tracks_torch_target = selected_tracks_torch_target.to(self.device)
                        selected_tracks_torch_output = selected_tracks_torch_output.to(self.device)

                        if shuffle:
                            target, pix_target, ticks_list_targ = all_sim(self.sim_target, selected_tracks_torch_target, self.track_fields,
                                                                          event_id_map, unique_eventIDs,
                                                                          return_unique_pix=True)
                            embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                        else:
                            # Simulate target and store them
                            if epoch == 0:
                                #No need to store gradients for forward-only pass
                                with torch.no_grad():
                                    #Update chunk sizes based on memory calculations
                                    self.optimize_batch_memory(self.sim_target, selected_tracks_torch_target)

                                    target, pix_target, ticks_list_targ = all_sim(self.sim_target, selected_tracks_torch_target, self.track_fields,
                                                                                event_id_map, unique_eventIDs,
                                                                                return_unique_pix=True)
                                    embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)

                                torch.save(embed_target, 'target_' + self.out_label + '/batch' + str(i) + '_ev' + str(int(ev))+ '_target.pt')

                            else:
                                embed_target = torch.load('target_' + self.out_label + '/batch' + str(i) + '_ev' + str(int(ev))+ '_target.pt')

                        # Undo normalization (sim -> sim_physics)
                        for param in self.relevant_params_list:
                            setattr(self.sim_physics, param, normalize_param(getattr(self.sim_iter, param), param, scheme=self.norm_scheme, undo_norm=True))
                            logger.debug(f"{param} {getattr(self.sim_physics, param)}")

                        # Simulate and get output
                        #Update chunk sizes based on memory calculations
                        self.optimize_batch_memory(self.sim_physics, selected_tracks_torch_output)
                        output, pix_out, ticks_list_out = all_sim(self.sim_physics, selected_tracks_torch_output, self.track_fields,
                                                  event_id_map, unique_eventIDs,
                                                  return_unique_pix=True)

                        # Embed both output and target into "full" image space
                        embed_output = embed_adc_list(self.sim_physics, output, pix_out, ticks_list_out)

                        # Calc loss between simulated and target + backprop
                        loss = self.loss_fn(embed_output, embed_target, **self.loss_fn_kw)

                        # To be investigated -- sometimes we get nans. Avoid doing a step if so
                        if not loss.isnan():
                            loss_ev.append(loss)

                    # Backpropagate the parameter(s) per batch
                    if len(loss_ev) > 0:
                        loss_ev_mean = torch.mean(torch.stack(loss_ev))
                        loss_ev_mean.backward()
                        if self.fit_diffs:
                            nan_check = torch.tensor([getattr(self.sim_iter, param+"_diff").grad.isnan() for param in self.relevant_params_list]).sum()
                        else:
                            nan_check = torch.tensor([getattr(self.sim_iter, param).grad.isnan() for param in self.relevant_params_list]).sum()
                        if nan_check == 0:
                            for param in self.relevant_params_list:
                                if self.fit_diffs:
                                    self.training_history[param+"_grad"].append(getattr(self.sim_iter, param+"_diff").grad.item())
                                else:
                                    self.training_history[param+"_grad"].append(getattr(self.sim_iter, param).grad.item())
                            if self.max_clip_norm_val is not None:
                                if self.fit_diffs:
                                    torch.nn.utils.clip_grad_norm_([getattr(self.sim_iter, param+"_diff") for param in self.relevant_params_list],
                                                               self.max_clip_norm_val)
                                else:
                                    torch.nn.utils.clip_grad_norm_([getattr(self.sim_iter, param) for param in self.relevant_params_list],
                                                               self.max_clip_norm_val)
                            self.optimizer.step()
                            losses_batch.append(loss_ev_mean.item())
                            self.training_history['losses_iter'].append(loss_ev_mean.item())
                            for param in self.relevant_params_list:
                                self.training_history[param+"_iter"].append(normalize_param(getattr(self.sim_iter, param).item(), 
                                                                                            param, scheme=self.norm_scheme, undo_norm=True))

                        else:
                            if len(self.training_history['losses_iter']) > 0:
                                self.training_history['losses_iter'].append(self.training_history['losses_iter'][-1])
                                for param in self.relevant_params_list:
                                    self.training_history[param+"_iter"].append(self.training_history[param+"_iter"][-1])
                                    self.training_history[param+"_grad"].append(self.training_history[param+"_grad"][-1])
                            else:
                                self.training_history['losses_iter'].append(0.)
                                for param in self.relevant_params_list:
                                    self.training_history[param+"_iter"].append(self.training_history[param+"_init"][0])
                                    self.training_history[param+"_grad"].append(0.)

                            logger.warning(f"Got {nan_check} gradients with a NaN value!")

                    else:
                        if len(self.training_history['losses_iter']) > 0:
                            self.training_history['losses_iter'].append(self.training_history['losses_iter'][-1])
                            for param in self.relevant_params_list:
                                self.training_history[param+"_iter"].append(self.training_history[param+"_iter"][-1])
                                self.training_history[param+"_grad"].append(self.training_history[param+"_grad"][-1])
                        else:
                            self.training_history['losses_iter'].append(0.)
                            for param in self.relevant_params_list:
                                self.training_history[param+"_iter"].append(self.training_history[param+"_init"][0])
                                self.training_history[param+"_grad"].append(0.)

                    if iterations is not None:
                        if total_iter % print_freq == 0:
                            for param in self.relevant_params_list:
                                logger.info(f"{param} {getattr(self.sim_physics,param).item()}")
                            
                        if total_iter % save_freq == 0:
                            with open(f'fit_result/history_{param}_iter{total_iter}_{self.out_label}.pkl', "wb") as f_history:
                                pickle.dump(self.training_history, f_history)

                            if os.path.exists(f'fit_result/history_{param}_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                os.remove(f'fit_result/history_{param}_iter{total_iter-save_freq}_{self.out_label}.pkl') 

                    total_iter += 1
                    pbar.update(1)
                    
                    if iterations is not None:
                        if total_iter >= iterations:
                            break

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Print out params at each epoch
                if epoch % print_freq == 0 and iterations is None:
                    for param in self.relevant_params_list:
                        logger.info(f"{param} {getattr(self.sim_physics,param).item()}")

                # Keep track of training history
                for param in self.relevant_params_list:
                    self.training_history[param].append(normalize_param(getattr(self.sim_iter, param).item(), param, scheme=self.norm_scheme, undo_norm=True))
                if len(losses_batch) > 0:
                    self.training_history['losses'].append(np.mean(losses_batch))

                # Save history in pkl files
                n_steps = len(self.training_history[param])
                if n_steps % save_freq == 0 and iterations is None:
                    with open(f'fit_result/history_{param}_epoch{n_steps}_{self.out_label}.pkl', "wb") as f_history:
                        pickle.dump(self.training_history, f_history)
                    if os.path.exists(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl'):
                        os.remove(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl') 

        if iterations is None:
            with open(f'fit_result/history_{param}_{self.out_label}.pkl', "wb") as f_history:
                pickle.dump(self.training_history, f_history)
                if os.path.exists(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl'):
                    os.remove(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl')

    def loss_scan_batch(self, dataloader, param_range=None, n_steps=10, shuffle=False, save_freq=5, print_freq=1):
        """
        Called by loss_landscape.py
        Old script that gets both loss and gradient, iterating over a parameter range
        Can be used to make gradient plot
        """
        
        if len(self.relevant_params_list) > 1:
            raise NotImplementedError("Can't do loss scan for more than one variable at a time!")

        param = self.relevant_params_list[0]
        if param_range is None:
            param_range = [ranges[param]['down'], ranges[param]['up']]
        param_vals = torch.linspace(param_range[0], param_range[1], n_steps)

        # make a folder for the pixel target
        if os.path.exists(f'target_{param}_batch'):
            shutil.rmtree(f'target_{param}_batch', ignore_errors=True)
        os.makedirs(f'target_{param}_batch')

        # make a folder for the output
        if os.path.exists(f'result_{param}'):
            shutil.rmtree(f'result_{param}', ignore_errors=True)
        os.makedirs(f'result_{param}')

        # The scanning loop
        with tqdm(total=len(param_vals)*len(dataloader)) as pbar:
            for i, selected_tracks_bt_torch in enumerate(dataloader):
                # Losses for each batch
                scan_losses = []
                scan_grads = []

                # Get rid of the extra dimension and padding elements for the loaded data
                selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                selected_tracks_bt_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("dx")] > 0]
                event_id_map, unique_eventIDs = get_id_map(selected_tracks_bt_torch, self.track_fields, self.device)

                # set up target per batch
                for ev in unique_eventIDs:
                    logger.info("batch: " + str(i) + '; ev:' + str(int(ev)))
                    selected_tracks_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("eventID")] == ev]
                    selected_tracks_torch = selected_tracks_torch.to(self.device)

                    if shuffle:
                        target, pix_target, ticks_list_targ = all_sim(self.sim_target, selected_tracks_torch, self.track_fields,
                                                                      event_id_map, unique_eventIDs,
                                                                      return_unique_pix=True)
                        embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                    else:
                        # Simulate target and store them
                        if os.path.exists(f'target_{param}_batch/batch' + str(i) + '_ev' + str(int(ev))+ '_target.pt'):
                            embed_target = torch.load(f'target_{param}_batch/batch' + str(i) + '_ev' + str(int(ev))+ '_target.pt')
 
                        else:
                            target, pix_target, ticks_list_targ = all_sim(self.sim_target, selected_tracks_torch, self.track_fields,
                                                                          event_id_map, unique_eventIDs,
                                                                          return_unique_pix=True)
                            embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)

                            torch.save(embed_target, f'target_{param}_batch/batch' + str(i) + '_ev' + str(int(ev))+ '_target.pt')


                for run_no, param_val in enumerate(param_vals):
                    setattr(self.sim_iter, param, param_val/ranges[param]['nom'])
                    self.sim_iter.track_gradients([param])

                    loss_ev_per_val = []
                    # Calculate loss per event
                    for ev in unique_eventIDs:

                        # Undo normalization (sim -> sim_physics)
                        for param in self.relevant_params_list:
                            setattr(self.sim_physics, param, getattr(self.sim_iter, param)*ranges[param]['nom'])
                            if run_no % print_freq == 0:
                                logger.info(f"{param}, {getattr(self.sim_physics, param)}")

                        # Simulate and get output
                        output, pix_out, ticks_list_out = all_sim(self.sim_physics, selected_tracks_torch, self.track_fields,
                                                  event_id_map, unique_eventIDs,
                                                  return_unique_pix=True)

                        # Embed both output and target into "full" image space
                        embed_output = embed_adc_list(self.sim_physics, output, pix_out, ticks_list_out)

                        # Calc loss between simulated and target + backprop
                        loss = self.loss_fn(embed_output, embed_target, **self.loss_fn_kw)

                        # To be investigated -- sometimes we get nans. Avoid doing a step if so
                        if not loss.isnan():
                            loss_ev_per_val.append(loss)
                        else:
                            logger.warning("Got NaN as loss!")

                    # Average out the loss in different events per batch
                    if len(loss_ev_per_val) > 0:
                        loss_ev_mean = torch.mean(torch.stack(loss_ev_per_val))
                        loss_ev_mean.backward()
                        scan_losses.append(loss_ev_mean.item())
                        scan_grads.append(getattr(self.sim_iter, param).grad.item())

                    else:
                        scan_losses.append(np.nan)
                        scan_grads.append(np.nan)


                    # store the scan outcome
                    if run_no % save_freq == 0:
                        recording = {'param' : param,
                                     'param_vals': param_vals,
                                     'norm_factor' : ranges[param]['nom'],
                                     'target_val' : getattr(self.sim_target, param),
                                     'losses' : scan_losses,
                                     'grads' : scan_grads }

                        outname = f"result_{param}/loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}_{run_no}"
                        with open(outname+".pkl", "wb") as f:
                            pickle.dump(recording, f)
                        if os.path.exists(f'result_{param}/loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}_{run_no-save_freq}.pkl'):
                            os.remove(f'result_{param}/loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}_{run_no-save_freq}.pkl')

                    pbar.update(1)

                recording = {'param' : param,
                             'param_vals': param_vals,
                             'norm_factor' : ranges[param]['nom'],
                             'target_val' : getattr(self.sim_target, param),
                             'losses' : scan_losses,
                             'grads' : scan_grads }
    
                outname = f"result_{param}/loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}"
                with open(outname+".pkl", "wb") as f:
                    pickle.dump(recording, f)
                if os.path.exists(f'result_{param}/loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}_{len(param_vals)-save_freq}.pkl'):
                    os.remove(f'result_{param}/loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}_{len(param_vals)-save_freq}.pkl')

            # average the loss and grad
            all_scan_losses = []
            all_scan_grads = []
            for i in range(len(dataloader)):
                #with open(f'loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}.pkl', 'rb') as pkl_file:
                history = pickle.load(open(f'result_{param}/loss_scan_batch{i}_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}.pkl', "rb"))
                all_scan_losses.append(np.array(history['losses']))
                all_scan_grads.append(np.array(history['grads']))
            scan_losses_mean = np.mean(np.array(all_scan_losses), axis=0)
            scan_grads_mean = np.mean(np.array(all_scan_grads), axis=0)

            recording = {'param' : param,
                         'param_vals': param_vals,
                         'norm_factor' : ranges[param]['nom'],
                         'target_val' : getattr(self.sim_target, param),
                         'losses' : scan_losses_mean,
                         'grads' : scan_grads_mean }

            outname = f"result_{param}/loss_scan_mean_{param}_{param_vals[0]:.02f}_{param_vals[-1]:.02f}"
            with open(outname+".pkl", "wb") as f:
                pickle.dump(recording, f)

        return recording, outname
    
    def scan_batch_loss(self, dataloader, shuffle=False, save_freq=5, print_freq=1, seed=None, data_seed=None, seed_init=None):
        """
        Called by loss_landscape.py
         - Save only the loss values without iterating parameters through gradient descent
         - calculates loss from target with loss from initial value for parameter, across all batches
        """
        # make a folder for the pixel target
        if not os.path.exists('fit_result'):
            os.mkdir('fit_result')
            # use os.makedirs if path has multiple directories
        
        # set initial values in recording
        recording = {}
        if seed:
            recording['seed'] = seed
        if data_seed:
            recording['data_seed'] = data_seed
        if seed_init:
            recording['seed_init'] = seed_init
        for param in self.relevant_params_list:
            recording[param+"_iter"] = getattr(self.sim_physics, param)
            recording[param+"_target"] = getattr(self.sim_target, param)

        # The scanning loop
        losses_batch = [] # losses per batch
        last_i = 0
        infinite_loss = ()
        with tqdm(total=len(dataloader)) as pbar:
            for i, selected_tracks_bt_torch in enumerate(dataloader):
                # Get rid of the extra dimension and padding elements for the loaded data
                selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                selected_tracks_bt_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("dx")] > 0]
                event_id_map, unique_eventIDs = get_id_map(selected_tracks_bt_torch, self.track_fields, self.device)

                losses_ev = [] # losses per event
                for ev in unique_eventIDs:
                    #logger.info("batch: " + str(i) + '; ev:' + str(int(ev)))
                    selected_tracks_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("eventID")] == ev]
                    selected_tracks_torch = selected_tracks_torch.to(self.device)

                    # set up target per event
                    target, pix_target, ticks_list_targ = all_sim(self.sim_target, selected_tracks_torch, self.track_fields,
                                                                          event_id_map, unique_eventIDs,
                                                                          return_unique_pix=True)
                    embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                    # Update chunk sizes based on memory calculations
                    self.optimize_batch_memory(self.sim_physics, selected_tracks_torch)
                    # Simulate and get output
                    # only need to update sim_physics iter if params are actually changing
                    output, pix_out, ticks_list_out = all_sim(self.sim_physics, selected_tracks_torch, self.track_fields,
                                                              event_id_map, unique_eventIDs,
                                                              return_unique_pix=True)
                    # Embed both output and target into "full" image space
                    embed_output = embed_adc_list(self.sim_physics, output, pix_out, ticks_list_out)
                    # Calc loss between simulated and target
                    loss = self.loss_fn(embed_output, embed_target, **self.loss_fn_kw)

                    # To be investigated -- sometimes we get nans. Avoid adding to event losses if so
                    if not loss.isnan(): 
                        if loss.isinf():
                            logger.warning("Got infinite loss! batch: " + str(i) + "; ev:" + str(int(ev)))
                        losses_ev.append(loss)
                    else:
                        logger.warning("Got NaN as loss! batch: " + str(i) + "; ev:" + str(int(ev)))

                # Average out the loss in different events per batch
                if len(losses_ev) > 0:
                    loss_ev_mean = torch.mean(torch.stack(losses_ev))
                    losses_batch.append(loss_ev_mean.item())
                else:
                    losses_batch.append(np.nan)

                # store the losses per batch
                if i % save_freq == 0:
                    recording['losses_iter'] = losses_batch
                    outname = f'fit_result/losses_batch{i}_{self.out_label}.pkl'
                    with open(outname, "wb") as f:
                        pickle.dump(recording, f)
                    if os.path.exists(f'fit_result/losses_batch{last_i}_{self.out_label}.pkl'):
                        os.remove(f'fit_result/losses_batch{last_i}_{self.out_label}.pkl')
                    last_i = i

                pbar.update(1)
        # store the losses one final time
        recording['losses_iter'] = losses_batch
        outname = f'fit_result/losses_batch{i}_{self.out_label}.pkl'
        with open(outname, "wb") as f:
            pickle.dump(recording, f)
        if os.path.exists(f'fit_result/losses_batch{last_i}_{self.out_label}.pkl'):
            os.remove(f'fit_result/losses_batch{last_i}_{self.out_label}.pkl')
        
        return recording, outname
