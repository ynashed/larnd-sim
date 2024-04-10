import os, sys
larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .utils import get_id_map, all_sim, embed_adc_list, calc_loss, calc_soft_dtw_loss
from .ranges import ranges
from larndsim.sim_jax import simulate, params_loss
from larndsim.consts_jax import build_params_class, load_detector_properties
import logging
import torch
import optax
import jax
import jax.numpy as jnp
from jax import value_and_grad

from tqdm import tqdm

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
    
def extract_relevant_params(params, relevant):
    return {par: getattr(params, par) for par in relevant}

def update_params(params, update):
    return params.replace(**{key: getattr(params, key) + val for key, val in update.items()})

class ParamFitter:
    def __init__(self, relevant_params, track_fields, track_chunk, pixel_chunk,
                 detector_props, pixel_layouts, load_checkpoint = None,
                 lr=None, optimizer=None, lr_scheduler=None, lr_kw=None, 
                 loss_fn=None, readout_noise_target=True, readout_noise_guess=False, 
                 out_label="", norm_scheme="divide", max_clip_norm_val=None, optimizer_fn="Adam",
                #  fit_diffs=False,
                 no_adc=False, shift_no_fit=[], link_vdrift_eField=False, batch_memory=None, skip_pixels = False,
                 set_target_vals=[], vary_init=False, seed_init=30,
                 config = {}):
        if optimizer_fn == "Adam":
            self.optimizer_fn = optax.adam
        elif optimizer_fn == "SGD":
            self.optimizer_fn = optax.sgd
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

        # self.fit_diffs = fit_diffs
        # If you have access to a GPU, sim works trivially/is much faster
        #TODO: See how to change/take care of CUDA availability with JAX
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
        # self.sim_target = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=readout_noise_target, skip_pixels=self.skip_pixels)
        # self.sim_target.load_detector_properties(detector_props, pixel_layouts)
        #TODO: take care of link_vdrift_eField

        # Simulation object for iteration -- this is where gradient updates will happen
        # self.sim_iter = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=readout_noise_guess, skip_pixels=self.skip_pixels)
        # self.sim_iter.load_detector_properties(detector_props, pixel_layouts)

        # Normalize parameters to init at 1, or random, or set to checkpointed values
        
        Params = build_params_class(self.relevant_params_list)
        ref_params = load_detector_properties(Params, detector_props, pixel_layouts)

        initial_params = {}

        for param in self.relevant_params_list:
            if is_continue:
                initial_params[param] = history[param][-1]
            else:
                if vary_init:
                    logger.info("Running with random initial guess")
                    init_val = np.random.uniform(low=ranges[param]['down'], 
                                                  high=ranges[param]['up'])
                    initial_params[param] = init_val

        self.current_params = ref_params.replace(**initial_params)
        self.params_normalization = ref_params.replace(**initial_params)
        self.norm_params = ref_params.replace(**{key: 1. for key in self.relevant_params_list})
         
        # # Placeholder simulation -- parameters will be set by un-normalizing sim_iter
        # self.sim_physics = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=readout_noise_guess, skip_pixels=self.skip_pixels)
        # self.sim_physics.load_detector_properties(detector_props, pixel_layouts)

        # for param in self.relevant_params_list:
        #     setattr(self.sim_physics, param, normalize_param(getattr(self.sim_iter, param), param, scheme=self.norm_scheme, undo_norm=True))

        # Keep track of gradients in sim_iter
        # self.sim_iter.track_gradients(self.relevant_params_list, fit_diffs=self.fit_diffs)

        self.learning_rates = {}
        #TODO: Only step the learning rate at each epoch!!!

        if lr_scheduler is not None and lr_kw is not None:
            lr_scheduler_fn = getattr(optax, lr_scheduler)
            logger.info(f"Using learning rate scheduler {lr_scheduler}")
        else:
            lr_scheduler_fn = optax.constant_schedule
            lr_kw = {}

        if self.relevant_params_dict is None:
            if lr is None:
                raise ValueError("Need to specify lr for params")
            else:
                self.learning_rates = {par: lr_scheduler_fn(lr, **lr_kw) for par in self.relevant_params_list}
        else:
            self.learning_rates = {key: lr_scheduler_fn(float(value), **lr_kw) for key, value in self.relevant_params_dict.items()}
        
        # Set up optimizer -- can pass in directly, or construct as SGD from relevant params and/or lr

        if optimizer is None:
            self.optimizer = optax.chain(
                optax.clip(0.1),
                optax.multi_transform({key: self.optimizer_fn(value) for key, value in self.learning_rates.items()},
                            {key: key for key in self.relevant_params_list})
            )
        else:
            raise ValueError("Passing directly optimizer is not supported")
            # self.optimizer = optimizer
        
        self.opt_state = self.optimizer.init(extract_relevant_params(self.norm_params, self.relevant_params_list))

        #TODO: Modify this part
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

            self.loss_fn_kw = {
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
                self.training_history[param + '_init'] = [getattr(self.current_params, param)]
                # self.training_history[param + '_lr'] = [lr_dict[param]]
            for param in self.shift_no_fit:
                self.training_history[param + '_target'] = []

            self.training_history['losses'] = []
            self.training_history['losses_iter'] = []
            self.training_history['norm_scheme'] = self.norm_scheme
        #     self.training_history['fit_diffs'] = self.fit_diffs
            self.training_history['optimizer_fn_name'] = self.optimizer_fn_name

        self.training_history['config'] = config

    def clip_values(self, mini=0.01, maxi=100):
        cur_norm_values = extract_relevant_params(self.norm_params, self.relevant_params_list)
        cur_norm_values = {key: max(mini, min(maxi, val)) for key, val in cur_norm_values.items()}
        self.norm_params = self.norm_params.replace(**cur_norm_values)

    def update_params(self):
        self.current_params = self.norm_params.replace(**{key: getattr(self.norm_params, key)*getattr(self.params_normalization, key) for key in self.relevant_params_list})

    def make_target_sim(self, seed=2, fixed_range=None):
        np.random.seed(seed)
        logger.info("Constructing target param simulation")

        self.target_params = {}

        if self.target_val_dict is not None:
            if set(self.relevant_params_list + self.shift_no_fit) != set(self.target_val_dict.keys()):
                logger.debug(set(self.relevant_params_list + self.shift_no_fit))
                logger.debug(set(self.target_val_dict.keys()))
                raise ValueError("Must specify all parameters if explicitly setting target")

            logger.info("Explicitly setting targets:")
            for param in self.target_val_dict.keys():
                param_val = self.target_val_dict[param]
                logger.info(f'{param}, target: {param_val}, init {getattr(self.current_params, param)}')    
                self.target_params[param] = param_val
        else:
            for param in self.relevant_params_list + self.shift_no_fit:
                if fixed_range is not None:
                    param_val = np.random.uniform(low=ranges[param]['nom']*(1.-fixed_range), 
                                                high=ranges[param]['nom']*(1.+fixed_range))
                else:
                    param_val = np.random.uniform(low=ranges[param]['down'], 
                                                high=ranges[param]['up'])

                logger.info(f'{param}, target: {param_val}, init {getattr(self.current_params, param)}')    
                self.target_params[param] = param_val
        self.target_params = self.current_params.replace(**self.target_params)

    
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
                self.training_history[param].append(getattr(self.current_params, param))
                self.training_history[param+'_target'].append(getattr(self.target_params, param))
            if len(self.training_history[param+"_iter"]) == 0:
                self.training_history[param+"_iter"].append(getattr(self.current_params, param))
        for param in self.shift_no_fit:
            if len(self.training_history[param+'_target']) == 0:
                self.training_history[param+'_target'].append(getattr(self.target_params, param))

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
                    # self.optimizer.zero_grad()

                    # Get rid of the extra dimension and padding elements for the loaded data
                    # selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                    # selected_tracks_bt_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("dx")] > 0]
                    # event_id_map, unique_eventIDs = get_id_map(selected_tracks_bt_torch, self.track_fields, self.device)
                    
                    #Convert torch tracks to jax
                    selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                    selected_tracks = jax.device_put(selected_tracks_bt_torch.numpy())

                    #Simulate the output for the whole batch
                    loss_ev = []

                    # selected_tracks_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("eventID")] == ev]
                    # selected_tracks_torch = selected_tracks_torch.to(self.device)

                    #Simulating the reference during the first epoch
                    fname = 'target_' + self.out_label + '/batch' + str(i) + '_target.npz'
                    if epoch == 0:
                        ref_adcs, ref_unique_pixels, ref_ticks = simulate(self.target_params, selected_tracks, self.track_fields)
                        # embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                        #Saving the target for the batch
                        #TODO: See if we have to do this for each event
                        
                        with open(fname, 'wb') as f:
                            jnp.savez(f, adcs=ref_adcs, unique_pixels=ref_unique_pixels, ticks=ref_ticks)
                    else:
                        #Loading the target
                        with open(fname, 'rb') as f:
                            loaded = jnp.load(f)
                            ref_adcs = loaded['adcs']
                            ref_unique_pixels = loaded['unique_pixels']
                            ref_ticks = loaded['ticks']

                    # Simulate and get output
                    (loss_val, aux), grads = value_and_grad(params_loss, (0), has_aux = True)(self.current_params, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=0)

                    print(f"Loss: {loss_val} ; ADC loss: {aux['adc_loss']} ; Time loss: {aux['time_loss']}")

                    updates, self.opt_state = self.optimizer.update(extract_relevant_params(grads, self.relevant_params_list), self.opt_state)
                    self.current_params = update_params(self.norm_params, updates)
                    self.norm_params = update_params(self.norm_params, updates)
                    self.clip_values()
                    self.update_params()


                    for param in self.relevant_params_list:
                        self.training_history[param+"_grad"].append(getattr(grads, param))
                    #TODO: Add norm clipping in the optimizer
                    # if self.max_clip_norm_val is not None:
                    #     if self.fit_diffs:
                    #         torch.nn.utils.clip_grad_norm_([getattr(self.sim_iter, param+"_diff") for param in self.relevant_params_list],
                    #                                     self.max_clip_norm_val)
                    #     else:
                    #         torch.nn.utils.clip_grad_norm_([getattr(self.sim_iter, param) for param in self.relevant_params_list],
                    #                                     self.max_clip_norm_val)
                    self.training_history['losses_iter'].append(loss_val)
                    for param in self.relevant_params_list:
                        self.training_history[param+"_iter"].append(getattr(self.current_params, param))

                    #     else:
                    #         if len(self.training_history['losses_iter']) > 0:
                    #             self.training_history['losses_iter'].append(self.training_history['losses_iter'][-1])
                    #             for param in self.relevant_params_list:
                    #                 self.training_history[param+"_iter"].append(self.training_history[param+"_iter"][-1])
                    #                 self.training_history[param+"_grad"].append(self.training_history[param+"_grad"][-1])
                    #         else:
                    #             self.training_history['losses_iter'].append(0.)
                    #             for param in self.relevant_params_list:
                    #                 self.training_history[param+"_iter"].append(self.training_history[param+"_init"][0])
                    #                 self.training_history[param+"_grad"].append(0.)

                    #         logger.warning(f"Got {nan_check} gradients with a NaN value!")

                    # else:
                    #     if len(self.training_history['losses_iter']) > 0:
                    #         self.training_history['losses_iter'].append(self.training_history['losses_iter'][-1])
                    #         for param in self.relevant_params_list:
                    #             self.training_history[param+"_iter"].append(self.training_history[param+"_iter"][-1])
                    #             self.training_history[param+"_grad"].append(self.training_history[param+"_grad"][-1])
                    #     else:
                    #         self.training_history['losses_iter'].append(0.)
                    #         for param in self.relevant_params_list:
                    #             self.training_history[param+"_iter"].append(self.training_history[param+"_init"][0])
                    #             self.training_history[param+"_grad"].append(0.)

                    if iterations is not None:
                        if total_iter % print_freq == 0:
                            for param in self.relevant_params_list:
                                logger.info(f"{param} {getattr(self.current_params,param)} {getattr(grads, param)} {updates[param]}")
                            
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

        #         # Print out params at each epoch
        #         if epoch % print_freq == 0 and iterations is None:
        #             for param in self.relevant_params_list:
        #                 logger.info(f"{param} {getattr(self.sim_physics,param).item()}")

        #         # Keep track of training history
        #         for param in self.relevant_params_list:
        #             self.training_history[param].append(normalize_param(getattr(self.sim_iter, param).item(), param, scheme=self.norm_scheme, undo_norm=True))
        #         if len(losses_batch) > 0:
        #             self.training_history['losses'].append(np.mean(losses_batch))

        #         # Save history in pkl files
        #         n_steps = len(self.training_history[param])
        #         if n_steps % save_freq == 0 and iterations is None:
        #             with open(f'fit_result/history_{param}_epoch{n_steps}_{self.out_label}.pkl', "wb") as f_history:
        #                 pickle.dump(self.training_history, f_history)
        #             if os.path.exists(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl'):
        #                 os.remove(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl') 

        # if iterations is None:
        #     with open(f'fit_result/history_{param}_{self.out_label}.pkl', "wb") as f_history:
        #         pickle.dump(self.training_history, f_history)
        #         if os.path.exists(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl'):
        #             os.remove(f'fit_result/history_{param}_epoch{n_steps-save_freq}_{self.out_label}.pkl')

    def loss_scan_batch(self, dataloader, param_range=None, n_steps=10, shuffle=False, save_freq=5, print_freq=1):

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
