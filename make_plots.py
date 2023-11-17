#!/usr/bin/env python3

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

labels = {'Ab' : 'Ab',
          'kb' : 'kb [g/cm$^2$/MeV]',
          'lifetime' : 'lifetime [$\mu s$]' ,
          'long_diff' : 'long_diff [$cm^2/\mu s$]',
          'tran_diff' : 'tran_diff [$cm^2/\mu s$]',
          'vdrift' : 'vdrift [$cm/\mu s$]',
          'eField' : 'eField [kV/cm]'}

def main(config):
    for param in config.params:
        for count, seed in enumerate(config.seeds):
            try:
                fname = glob(f"history*_seed{seed}_{config.label}.pkl")[0]
            except:
                continue

            history = pickle.load(open(fname, "rb"))

            plt.plot(np.asarray(history[f"{param}_iter"]), c=f'C{count}')
            plt.plot([0, len(history[f"{param}_iter"])], [history[f'{param}_target']]*2, c=f'C{count}', ls='dashed')

        plt.ylabel(labels[param])
        plt.xlabel('Training Iteration')
        plt.tight_layout()
        plt.savefig(f'plot_{param}_{config.label}.{config.ext}', dpi=300)
        print(f"Saving plot to plot_{param}_{config.label}.{config.ext}")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="params", default=[], nargs="+",
                        help="List of parameters to plot.")
    parser.add_argument("--label", dest="label",
                        help="Label of pkl file.") 
    parser.add_argument("--seeds", dest="seeds", default=[], nargs="+",
                        help="List of target seeds to plot.") 
    parser.add_argument("--ext", dest="ext", default="pdf",
                        help="Image extension (e.g., pdf or png)") 

    args = parser.parse_args()
    main(args)