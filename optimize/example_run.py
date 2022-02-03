#!/usr/bin/env python3

from ParamFitter import ParamFitter

def main():
    param_fit = ParamFitter(['eField', 'lifetime'], lr=1e1) 
    param_fit.make_target_sim(seed=2)
    param_fit.load_data(filename='/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5')
    param_fit.fit(epochs=1)


if __name__ == "__main__":
    main()
