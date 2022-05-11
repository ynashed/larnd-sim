#!/usr/bin/env python3

from .test_utils import calc_forward, loss_fn

import torch
import os
import pickle

def test_backward():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    target = torch.load('tests/data/testSim_ep_out.pth').to(device)
    param_list = ['eField', 'lifetime', 'vdrift', 'tran_diff', 'long_diff', 'MeVToElectrons', 'Ab', 'kb']

    guess_path = 'tests/data/testSim_ep_guess.pth'
    param_path = 'tests/data/testSim_ep_param_grads.pkl'

    output_with_grads, sim = calc_forward(with_grad=True, param_list=param_list, shift=0.05, device=device)
    if os.path.exists(guess_path):
        check = torch.load(guess_path).to(device)
        torch.save(output_with_grads, 'tests/output/forward-test-output-guess.pth')
        assert torch.allclose(output_with_grads, check), f'Forward model output with grads differs! Max diff: {abs(output_with_grads-check).max()}'
    else:
        print("Saving new comparison file")
        torch.save(output_with_grads, guess_path)

    loss = loss_fn(output_with_grads, target)
    loss.backward()

    recording = {}
    recording['loss'] = loss.item()
    for param in param_list:
        recording[f'{param}_grad'] = getattr(sim, param).grad.item()

    with open('tests/output/backward-test-output.pkl', 'wb') as f:
        pickle.dump(recording, f)

    if os.path.exists(param_path):
        check_params = pickle.load(open(param_path, "rb"))
        for key in check_params.keys():
            assert torch.allclose(torch.tensor(check_params[key]), torch.tensor(recording[key])), f'{key} differs! Diff: {abs(check_params[key]-recording[key])}'
    else:
        print("Saving new comparison file")
        with open(param_path, 'wb') as f:
            pickle.dump(recording, f)

