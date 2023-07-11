from optimize.dataio import TracksDataset
from optimize.utils import all_sim, get_id_map, embed_adc_list

from larndsim.sim_with_grad import sim_with_grad

import torch
from torch.utils.data import DataLoader

import os
import shutil
from tqdm import tqdm

import argparse
    
def optimize_batch_memory(sim, tracks, track_fields) -> None:
    estimated_memory = sim.estimate_peak_memory(tracks, track_fields)
    chunk_size = int(32768 // estimated_memory)
    chunk_size = max(1, chunk_size) #Min value should be 1
    #logger.info(f"Initial maximum memory of {estimated_memory/1024:.2f}Gio. Setting pixel_chunk_size to {chunk_size} and expect a maximum memory of {chunk_size*estimated_memory/1024:.2f}Gio")
    sim.update_chunk_sizes(1, chunk_size)

def main(config):
    track_chunk = 1
    pixel_chunk = 1
    detector_props = "larndsim/detector_properties/module0.yaml"
    pixel_layouts = "larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml"
    input_file =  "/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5"

    dataset = TracksDataset(filename=input_file, ntrack=-1, max_batch_len=100, min_abs_segz_sel=15)
    track_fields = dataset.get_track_fields()
    device='cuda'

    dataloader = DataLoader(dataset,
                            shuffle=False, 
                            batch_size=1,
                            pin_memory=True, num_workers=2)
    
    sim_target = sim_with_grad(track_chunk=track_chunk, pixel_chunk=pixel_chunk, readout_noise=config.noise)
    sim_target.load_detector_properties(detector_props, pixel_layouts)
    sim_target.link_vdrift_eField = True
    
    for i in range(len(config.param_list)):
        val = config.param_vals[i]
        setattr(sim_target, config.param_list[i], val)
    
    # make a folder for the pixel target
    if os.path.exists(f"{config.save_name}_batch"):
        shutil.rmtree(f"{config.save_name}_batch", ignore_errors=True)
    os.makedirs(f"{config.save_name}_batch")

    with tqdm(total=len(dataloader)) as pbar:
        for i, selected_tracks_bt_torch in enumerate(dataloader):

            # Get rid of the extra dimension and padding elements for the loaded data
            selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
            selected_tracks_bt_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, track_fields.index("dx")] > 0]
            event_id_map, unique_eventIDs = get_id_map(selected_tracks_bt_torch, track_fields, device)

            for ev in unique_eventIDs:
                selected_tracks_torch = selected_tracks_bt_torch[selected_tracks_bt_torch[:, track_fields.index("eventID")] == ev]
                selected_tracks_torch = selected_tracks_torch.to(device)

                optimize_batch_memory(sim_target, selected_tracks_torch, track_fields)
                target, pix_target, ticks_list_targ = all_sim(sim_target, selected_tracks_torch, track_fields,
                                                                event_id_map, unique_eventIDs,
                                                                return_unique_pix=True)
                embed_target = embed_adc_list(sim_target, target, pix_target, ticks_list_targ)

                torch.save(embed_target, f'{config.save_name}_batch/batch' + str(i) + '_ev' + str(int(ev))+ '.pt')

            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="param_list", default=[], nargs="+",
                        help="List of parameters to vary. See consts_ep.py")
    parser.add_argument("--param-vals", dest="param_vals", default=[], nargs="+", type=float,
                        help="Parameter values.")
    parser.add_argument("--noise", dest="noise", default=False, action="store_true",
                        help="Add noise in sim.")
    parser.add_argument("--save-name", dest="save_name", default=None,
                        help="Save name for output files")
    args = parser.parse_args()
    main(args)