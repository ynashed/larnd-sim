#!/usr/bin/env python3

from .test_utils import calc_forward

import torch
import os

def test_forward():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    output_path = 'tests/data/testSim_ep_out.pth'
    outputs = calc_forward(device=device)
    if os.path.exists(output_path):
        check = torch.load(output_path).to(device)
        torch.save(outputs, 'tests/output/forward-test-output.pth')
        assert torch.allclose(outputs, check), f'Forward model output differs! Max diff: {abs(outputs-check).max()}'
    else:
        print("Saving new comparison file")
        torch.save(outputs, output_path)
