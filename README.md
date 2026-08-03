# Experimental Bayesian Model Updating Benchmark in UncertaintyQuantification.jl

Code and data for the benchmark study in  
*Bayesian Model Updating of Structural Dynamics: A 4-DOF Experimental Benchmark Using UncertaintyQuantification.jl* (EURODYN 2026 submission).


This repository demonstrates Bayesian model updating for a 4-DOF structural dynamics benchmark using the [UncertaintyQuantification.jl](https://github.com/juliauq/UncertaintyQuantification.jl) package.

## Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Note: The workflow in [`main_updating.jl`](main_updating.jl) also uses PlotMatrix for visualization, available at https://github.com/lukasfritsch/PlotMatrix.jl.

## Run The Benchmark Script

```bash
julia --project=. main_updating.jl
```

The script [`main_updating.jl`](main_updating.jl):

- loads measurement data from `measurements_br.txt` and `measurements_de.txt`
- evaluates the 4-DOF mechanical model in `mechmodels.jl`
- computes likelihoods defined in `likelihood.jl`
- performs Bayesian updating with `UncertaintyQuantification.jl` methods
- generates plots and exports CSV files to timestamped folders inside results

## Inference Methods

The benchmark compares the following methods (implemented in [UncertaintyQuantification.jl](https://github.com/juliauq/UncertaintyQuantification.jl)):

- Single-Component Metropolis-Hastings
- Transitional Markov Chain Monte Carlo (TMCMC)
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP)
- Laplace Approximation
- Transport Map Bayesian Inference

For more information, see: [Bayesian Updating documentation](https://juliauq.github.io/UncertaintyQuantification.jl/stable/manual/bayesianupdating)

## Folder Structure

```text
.
├── main_updating.jl
├── data
│   ├── measurements_br.txt
│   └── measurements_de.txt
├── results
└── src
    ├── mechmodels.jl
    ├── datahandling.jl
    ├── likelihood.jl
    ├── transformations.jl
    └── postprocess.jl
```

## Citation
If you use the data in your research, please cite using the information provided at `Cite this repository` (or see [`CITATION.cff`](CITATION.cff)).