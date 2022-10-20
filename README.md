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
- Quenching: contained in `quenching_ep.py`. 
- Drifting: contained in `drifting_ep.py`.
- Pixelization: contained in `pixels_from_track_ep.py`
- Charge/current calculation: contained in `detsim_ep.py`
- Electronics simulation: contained in `fee_ep.py`.

## What does differentiable mean?
Let's think of our simulator as some function $f(x,\theta)$ which maps from dEdx track segments ($x$) to the corresponding 
pixel readout $f(x)$, where $\theta$ are some set of parameters of the simulation. We call this function $f$ our "forward model". 
If this model is differentiable, this means that we can calculate $\nabla_{\theta} f(x, \theta)$ (and/or $\nabla_{x} f(x, \theta)$), 
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
- Take some data, which we assume to be from $f(x^{\\*}, \theta^{\\*})$, with $x^{\\*}$, $\theta^{\\*}$ the "true" values seen in real life.
- Assuming we know (or can estimate) $x^{\\*}$, simulate some data with a set of parameters $\theta_{i}$.
- Compare the simulation output $f(x^{\\*}, \theta_{i})$, with data $f(x^{\\*}, \theta^{\\*})$ using some _loss function_
- Update parameters $\theta_{i}$ via gradient descent to minimize the chosen loss, i.e.,
$$
\theta_{i} \rightarrow \theta_{i} - \eta\cdot\nabla_{\theta} f(x^{\\*}, \theta_{i})
$$
where $\eta$ is some _learning rate_ that controls how big of a step we take. This procedure (falling into an "analysis-by-synthesis" framework)
can be repeated until convergence is reached.

Note that this is one particular application -- reconstruction of inputs (via a similar procedure with $\nabla_{x} f(x, \theta)$) is a very related 
process. However the differentiability also allows for training of neural networks in conjunction with physics simulation (as gradients with respect 
to neural network weights and biases may be passed all the way through). TL; DR, this can be useful!

## Code overview and how to run a fit


