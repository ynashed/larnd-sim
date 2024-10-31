import os, sys
larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .utils import get_id_map, all_sim, embed_adc_list
from .ranges import ranges
from larndsim.sim_jax import simulate, params_loss, loss, simulate_parametrized, params_loss_parametrized
from larndsim.consts_jax import build_params_class, load_detector_properties
from larndsim.softdtw_jax import SoftDTW
import logging
import torch
import optax
import jax
import jax.numpy as jnp
from jax import value_and_grad

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
                 set_target_vals=[], vary_init=False, seed_init=30, profile_gradient = False,
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

        self.profile_gradient = profile_gradient

        self.current_mode = config.mode
        self.electron_sampling_resolution = config.electron_sampling_resolution
        self.number_pix_neighbors = config.number_pix_neighbors
        self.signal_length = config.signal_length

        if self.current_mode == 'lut':
            self.lut_file = config.lut_file
            self.load_lut()
            self.number_pix_neighbors = 0


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
        ref_params = ref_params.replace(
            electron_sampling_resolution=self.electron_sampling_resolution,
            number_pix_neighbors=self.number_pix_neighbors,
            signal_length=self.signal_length,
            time_window=self.signal_length)

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
            self.loss_fn = loss
            self.loss_fn_kw = { }
            logger.info("Using space match loss")
        elif loss_fn == "SDTW":
            self.loss_fn = self.calc_sdtw

            loss_fn_kw = {
                            'gamma' : 1
                            }
            self.loss_fn_kw = {}
            self.dstw = SoftDTW(**loss_fn_kw)
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

    def load_lut(self):
        response = np.load(self.lut_file)
        extended_response = np.zeros((50, 50, 1891))
        extended_response[:45, :45, :] = response
        response = extended_response
        baseline = np.sum(response[:, :, :-self.signal_length+1], axis=-1)
        response = np.concatenate([baseline[..., None], response[..., -self.signal_length+1:]], axis=-1)
        self.response = response

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

        if self.profile_gradient:
            logger.info("Using the fitter in a gradient profile mode. The sampling will follow a regular grid.")
            nb_var_params = len(self.relevant_params_list)
            logger.info(f"{nb_var_params} parameters are to be scanned.")
            nb_steps = int(epochs**(1./nb_var_params))
            logger.info(f"Each parameter will be scanned with {nb_steps} steps")
            grids_1d = [list(range(nb_steps))]*nb_var_params
            steps_grids = np.meshgrid(*grids_1d)

            for i, param in enumerate(self.relevant_params_list):
                lower = ranges[param]['down']
                upper = ranges[param]['up']
                param_step = (upper - lower)/(nb_steps - 1)
                steps_grids[i] = steps_grids[i].astype(float)
                steps_grids[i] *= param_step
                steps_grids[i] += lower
                steps_grids[i] = steps_grids[i].ravel()

        # The training loop
        total_iter = 0
        with tqdm(total=pbar_total) as pbar:
            for epoch in range(epochs):
                # Losses for each batch -- used to compute epoch loss
                losses_batch=[]

                if self.profile_gradient:
                    new_param_values = {}
                    for i, param in enumerate(self.relevant_params_list):
                        new_param_values[param] = steps_grids[i][epoch]
                    logger.info(f"Stepping parameter values: {new_param_values}")
                    self.current_params = self.current_params.replace(**new_param_values)
                for i, selected_tracks_bt_torch in enumerate(dataloader):
                    # Zero gradients
                    # self.optimizer.zero_grad()

                    # Get rid of the extra dimension and padding elements for the loaded data
                    # selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                    # selected_tracks_bt_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, self.track_fields.index("dx")] > 0]
                    
                    #Convert torch tracks to jax
                    selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                    selected_tracks = jax.device_put(selected_tracks_bt_torch.numpy())

                    #Simulate the output for the whole batch
                    loss_ev = []

                    #Simulating the reference during the first epoch
                    fname = 'target_' + self.out_label + '/batch' + str(i) + '_target.npz'
                    if epoch == 0:
                        if self.current_mode == 'lut':
                            ref_adcs, ref_unique_pixels, ref_ticks = simulate(self.target_params, self.response, selected_tracks, self.track_fields)
                        else:
                            ref_adcs, ref_unique_pixels, ref_ticks = simulate_parametrized(self.target_params, selected_tracks, self.track_fields)
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
                    # loss_val, aux = params_loss(self.current_params, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=0, loss_fn=self.loss_fn, **self.loss_fn_kw)
                    # grads, aux = jax.grad(params_loss, (0), has_aux = True)(self.current_params, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=0, loss_fn=self.loss_fn, **self.loss_fn_kw)
                    if self.current_mode == 'lut':
                        (loss_val, aux), grads = value_and_grad(params_loss, (0), has_aux = True)(self.current_params, self.response, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=0, loss_fn=self.loss_fn, **self.loss_fn_kw)
                    else:
                        (loss_val, aux), grads = value_and_grad(params_loss_parametrized, (0), has_aux = True)(self.current_params, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=0, loss_fn=self.loss_fn, **self.loss_fn_kw)
                    print(f"Loss: {loss_val} ; ADC loss: {aux['adc_loss']} ; Time loss: {aux['time_loss']}")

                    if not self.profile_gradient:
                        updates, self.opt_state = self.optimizer.update(extract_relevant_params(grads, self.relevant_params_list), self.opt_state)
                        self.current_params = update_params(self.norm_params, updates)
                        self.norm_params = update_params(self.norm_params, updates)
                        #Clipping param values
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
                                # logger.info(f"{param} {getattr(self.current_params,param)} {getattr(grads, param)} {updates[param]}")
                                logger.info(f"{param} {getattr(self.current_params,param)} {getattr(grads, param)}")
                            
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

    #TODO: Finish this thing
    def calc_sdtw(self, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, fields):

        sorted_adcs = adcs[jnp.argsort(pixels)]
        sorted_ref = ref[jnp.argsort(pixels_ref)]

        # sorted_adcs = adcs.flatten()[jnp.argsort(ticks.flatten())]
        # sorted_ref = ref.flatten()[jnp.argsort(ticks_ref.flatten())]
        adc_loss = self.dstw.pairwise(sorted_adcs, sorted_ref)
        # adc_loss = adc_loss/len(sorted_adcs)/len(sorted_ref)
        time_loss = 0
        loss = adc_loss + time_loss
        aux = {
            'adc_loss': adc_loss,
            'time_loss': time_loss
        }

        return loss, aux