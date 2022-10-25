# larnd-sim 
![CI status](https://github.com/DUNE/larnd-sim/workflows/CI/badge.svg)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://dune.github.io/larnd-sim)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4582721.svg)](https://doi.org/10.5281/zenodo.4582721)

<img alt="larnd-sim" src="docs/logo.png" height="160">

This software aims to simulate a pixelated Liquid Argon Time Projection Chamber. This is a _differentiable_ version of a snapshot of 
[DUNE larnd-sim](https://github.com/DUNE/larnd-sim), and is (as closely as possible), a direct translation of that code. Rather 
than relying on explicit CUDA kernel parallelization, this code has been (a) vectorized and (b) translated into [EagerPy](https://github.com/jonasrauber/eagerpy), 
a backend agnostic framework allowing for the use of a variety of automatic differentiation libraries. The optimization code here is written in 
[PyTorch](https://pytorch.org/), and there are a few residual native PyTorch functions used throughout the main code.

For comparison, files from the non-differentiable version of larnd-sim have been kept, with the translated software denoted with a subscript 
`_ep` (e.g. `quenching.py` -> `quenching_ep.py`). Note that it is possible to run on both GPU and CPU with this code, with GPU offering 
considerable speed-ups due to the vectorization. However, because of the vectorization, the memory use is fairly high -- typical runs are 
done on the SLAC Shared Scientific Data Facility (SDF) using single NVIDIA Tesla A100s with 40GB of VRAM.

## Physics overview
Following the structure of the non-differentiable code, the simulation proceeds in stages, which can be seen in the structure 
of the `all_sim` function in `optimize/utils.py`. These stages are
- Quenching: contained in `quenching_ep.py`. Goes from energy deposits to number of electrons (after recombination). Uses Birks model 
for this calculation by default. Parameters: `Ab`, `kb`, `eField`.
- Drifting: contained in `drifting_ep.py`. Calculates properties of segments at the anode, after drifting through the detector. 
Parameters: `vdrift`, `tran_diff`, `long_diff`, `lifetime`.
- Pixelization: contained in `pixels_from_track_ep.py`. Finds pixel projection of tracks (Bresenham's line algorithm). `tran_diff` implicitly 
enters in determining max spread.
- Charge/current calculation: contained in `detsim_ep.py`. Calculates current induced on each pixel. Almost all parameters enter somewhere either 
implicitly or explicitly.  
- Electronics simulation: contained in `fee_ep.py`. Simulates electronics/triggering/converts current to ADC counts. This is where "noise" can enter.

## What does differentiable mean?
Let's think of our simulator as some function $f(x,\theta)$ which maps from dEdx track segments, $x$, to the corresponding 
pixel readout $f(x)$, where $\theta$ are some set of parameters of the simulation. We call this function $f$ our "forward model". 
If this model is differentiable, this means that we can calculate $\nabla_{\theta} f(x, \theta)$ and/or $\nabla_{x} f(x, \theta)$, 
the gradient of our forward model output with respect to its parameters, $\theta$, or inputs, $x$.

Doing such a calculation is made very efficient by a tool called _automatic differentiation_, the backbone of frameworks such as 
PyTorch. Full discussion of automatic differentiation is beyond the scope of this README. For the optimization here, we use 
_reverse mode automatic differentiation_, also known as backpropagation, which effectively uses the chain rule moving backward 
from the output of the forward model in order to exactly compute the desired derivatives.

It is important to note that not all operations in the standard larnd-sim are nicely differentiable, for instance discontinuous integer 
operations, thresholding, etc etc. In several places, we therefore do a _relaxation_ of these operations. A common example is to take a 
hard, step function cut, and replace it with a sigmoid function, which has similar behavior on the tails, but replaces a discontinuous jump 
with a smooth rise to terminal values (e.g. from 0 to 1). Such relaxations allow for the calculation of gradients, at the price of the accuracy 
of the simulation. We include a few such relaxations in the differentiable simulation code, notably:
- Interpolation between time steps in integrated current thresholding
- Relaxation of current model shape (truncated exponential)
- Integer ADC values -> floating point

Once we do have reasonable gradients, however, this opens up several possibilities. One of the major focuses of this work is the idea of 
_calibration_. This particularly focuses on the use of gradients with respect to parameters, $\nabla_{\theta} f(x, \theta)$. The general 
setup is:
- Take some data, which we assume to be from $f(x^\*, \theta^\*)$, with $x^\*$, $\theta^\*$ the "true" values seen in real life.
- Assuming we know (or can estimate) $x^\*$, simulate some data with a set of parameters $\theta_{i}$.
- Compare the simulation output $f(x^\*, \theta_{i})$, with data $f(x^\*, \theta^\*)$ using some _loss function_
- Update parameters $\theta_{i}$ via gradient descent to minimize the chosen loss, i.e.,
$$\theta_{i} \rightarrow \theta_{i} - \eta\cdot\nabla_{\theta} f(x^\*, \theta_{i})$$
where $\eta$ is some _learning rate_ that controls how big of a step we take. This procedure (falling into an "analysis-by-synthesis" framework)
can be repeated until convergence is reached.

Note that this is one particular application -- reconstruction of inputs, via a similar procedure with $\nabla_{x} f(x, \theta)$, is a very related 
process. Further, the differentiability also allows for training of neural networks in conjunction with physics simulation (as gradients with respect 
to neural network weights and biases may be passed all the way through). TL; DR, this can be useful!

## Fitting code overview
Most of the physics code is in `larndsim/`, while the optimization code is in `optimize/`. For the following, we assume that you have access to SLAC's 
SDF (so that you can use the corresponding resources and file paths). The heart of the user interface is the file `optimize/example_run.py`, 
which makes use of the `ParamFitter` class defined in `optimize/fit_params.py`, as well as the data handling of `optimize/dataio.py`. 

An image containing all required packages is located at:
`/sdf/group/neutrino/images/larndsim_latest.sif`

And we've been using a set of simulated muon data for tests:
`/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5`

This fitting framework is set up by default to run a "closure test" style fit. Namely, we:
- Simulate some "data" with a given set of parameters. This is the "target data".
- Initialize at some reasonable guess. For us, this is often a set of nominal parameter values.
- Run the gradient descent optimization procedure to update parameter guesses (starting from the initial value).

The goal of this closure test is to see if we can recover the target parameter values in an "ideal" case. To that 
end we've included a flag to turn off electronic noise (the only source of stochasticity in the detector simulation).

The simplest way to run your first fit is using an included batch script at `optimize/scripts/example_submit.sh`. Namely, 
on SDF, run:
```
sbatch optimize/scripts/example_submit.sh
```
from the top level `larnd-sim` directory.

This spawns 5 jobs with different randomly seeded targets within "reasonable" parameter ranges defined in `optimize/ranges.py`. 
It runs a fit to these targets on a set of 5 tracks in a single batch, using an Adam optimizer and [soft DTW](https://arxiv.org/abs/1703.01541) 
loss. This may take a bit to run. Note that this uses the `ml` partition and a single NVIDIA Tesla A100 for each job -- if you're not a member 
of the `ml` partition, you may need to change to something else (`shared`, e.g.).

As the jobs run, they store a bunch of information in a dict, saved in a pkl file -- this can then be used to plot/analyze results, e.g.
to make something like this plot: (to do, include)

The parameters of interest for fitting are (currently), in rough priority order:
- Ab
- kb
- tran_diff
- vdrift
- lifetime
- eField
- long_diff

Note that vdrift is "special" in that the dominant impact is in time info (and it has by far the largest effect there), so this is used 
explicitly in the fitting.

The example we had you run does a fit for a single parameter (Ab) on a limited set of tracks. Multiparameter fitting is possible by passing 
either:
- A space separated list of parameters (one learning rate used for all)
- A `.yaml` file defining the list of parameters and their associated learning rates, e.g.,

``` yaml
Ab: 1e-2
kb: 1e-1
```
will run a simultaneous (2D) fit in `Ab` and `kb` with learning rates 1e-2 and 1e-1 respectively.

There are several other configuration flags, please see the included help text for more information on those. 
