#!/usr/bin/env python3

from ParamFitter import ParamFitter
import argparse

def main(param_list, input_file, lr, epochs, seed):
    param_fit = ParamFitter(param_list, lr=lr) 
    param_fit.make_target_sim(seed=seed)
    param_fit.load_data(filename=input_file)
    param_fit.fit(epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="param_list", default=[], nargs="+", 
                        help="List of parameters to optimize. See consts_ep.py")
    parser.add_argument("--input-file", dest="input_file", default="",
                        help="Input data file")
    parser.add_argument("--lr", dest="lr", default=1e1, type=float,
                        help="Learning rate -- used for all params")
    parser.add_argument("--epochs", dest="epochs", default=100, type=int,
                        help="Number of epochs")
    parser.add_argument("--seed", dest="seed", default=2, type=int,
                        help="Random seed for target construction")
   
    args = parser.parse_args()
    main(args.param_list, args.input_file, args.lr, args.epochs, args.seed)
