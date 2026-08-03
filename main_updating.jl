using UncertaintyQuantification
using DataFrames
using LaTeXStrings
using LinearAlgebra
using CSV
using Dates
using Optim
using CairoMakie
using PlotMatrix # install from https://github.com/lukasfritsch/PlotMatrix.jl
using Random

# Set seed for reproduciability
Random.seed!(67)

include("src/mechmodels.jl")
include("src/datahandling.jl")
include("src/likelihood.jl")
include("src/transformations.jl")
include("src/postprocess.jl")

# Plot options
set_theme!(theme_latexfonts())
labels = [L"$\theta_1$ [-]", L"$\theta_2$ [-]", L"$\theta_3$ [-]", L"$\theta_4$ [-]", L"$\theta_5$ [-]", L"$\theta_6$ [-]", L"$\theta_7$ [-]", L"$\theta_8$ [-]"]
data_labels = [L"\biD^{(\mathrm{BR})}", L"\biD^{(\mathrm{DE})}"]
plot_size = (600, 600)
output_plot_size = (600, 400)

## Save results?
save_results = true
time_stamp = Dates.format(now(), "yyyymmdd_HHMMSS")

# Define UQ.jl parallel model
model = ParallelModel(
    df -> dampedfourDofmodel(prior_to_physical(df)), :f
)

# Define prior in the standard normal space Z ∼ N(0, 1)
prior = [
    RandomVariable(Normal(), :z1),
    RandomVariable(Normal(), :z2),
    RandomVariable(Normal(), :z3),
    RandomVariable(Normal(), :z4),
    RandomVariable(Normal(), :z5),
    RandomVariable(Normal(), :z6),
    RandomVariable(Normal(), :z7),
]

# Load the data
data_de = load_frequencies("data/measurements_de.txt")
data_br = load_frequencies("data/measurements_br.txt")
keep = [(!any(isnan, row)) for row in eachrow(data_de)]
data_de = data_de[keep, :]

# Define the likelihood
like_de = Model(df -> likelihood(df, data_de), :L)
like_br = Model(df -> likelihood(df, data_br), :L)

## Metropolis-Hastings
n = 10_000
burnin = 5_000

x0 = (z1=0., z2=0., z3=0., z4=0., z5=0., z6=0., z7=0.)

proposal = [
    Normal(0, 1),
    Normal(0, 1),
    Normal(0, 1),
    Normal(0, 0.01),
    Normal(0, 0.01),
    Normal(0, 0.01),
    Normal(0, 0.01)
]

mh = SingleComponentMetropolisHastings(proposal, x0, n, burnin)
mh_samples_de, α_de = bayesianupdating(logprior, df -> df.L, [model, like_de], mh)
mh_samples_br, α_br = bayesianupdating(logprior, df -> df.L, [model, like_br], mh)

## Transitional Markov Chain Monte Carlo
burnin = 3

tmcmc = TransitionalMarkovChainMonteCarlo(prior, n, burnin)
tmcmc_samples_de, _ = bayesianupdating(df -> df.L, [model, like_de], tmcmc)
tmcmc_samples_br, _ = bayesianupdating(df -> df.L, [model, like_br], tmcmc)

## Maximum Likelihood Estimation
x0 = zeros(7)

MLE = MaximumLikelihoodBayesian(prior, x0)
MLEEstimate_de = bayesianupdating(df -> df.L, [model, like_de], MLE)
MLEEstimate_physical_de = unique_df_physical(MLEEstimate_de)

MLEEstimate_br = bayesianupdating(df -> df.L, [model, like_br], MLE)
MLEEstimate_physical_br = unique_df_physical(MLEEstimate_br)
## Maximum A-Posteriori Estimation
MAP = MaximumAPosterioriBayesian(prior, x0)
MAPEstimate_de = bayesianupdating(df -> df.L, [model, like_de], MAP)
MAPEstimate_physical_de = unique_df_physical(MAPEstimate_de)

MAPEstimate_br = bayesianupdating(df -> df.L, [model, like_br], MAP)
MAPEstimate_physical_br = unique_df_physical(MAPEstimate_br)
## Laplace Estimation
LaplaceEstimator = LaplaceEstimateBayesian(prior, x0)
laplace_de = bayesianupdating(df -> df.L, [model, like_de], LaplaceEstimator)
laplace_br = bayesianupdating(df -> df.L, [model, like_br], LaplaceEstimator)

## Transport Map
T = PolynomialMap(7, 3, Normal(), ShiftedELU(), LinearizedHermiteBasis(), :no_mixed)
quad = LatinHypercubeWeights(200, 7)
tmb = TransportMapBayesian(prior, T, quad, AutoFiniteDiff(), LBFGS(),
    Optim.Options(show_trace=true, iterations=500, show_every=25), transformprior=false)

tm_de = bayesianupdating(df -> df.L, [model, like_de], deepcopy(tmb))
tm_br = bayesianupdating(df -> df.L, [model, like_br], deepcopy(tmb))

## Plot Samples
mh_plot = plotmatrix(unique_df_physical(mh_samples_br), unique_df_physical(mh_samples_de);
    dimlabels=labels, size=plot_size, xticklabelrotation=pi\2, legend_labels=data_labels,
    alphas=(0.8, 0.8, 0.8), ticklabelsize=9, markersize=2
)

tmcmc_plot = plotmatrix(unique_df_physical(tmcmc_samples_br), unique_df_physical(tmcmc_samples_de);
    dimlabels=labels, size=plot_size, xticklabelrotation=pi\2, legend_labels=data_labels,
    alphas=(0.8, 0.8, 0.8), ticklabelsize=9, markersize=2
)

tm_samples_br = sample(tm_br, n)
tm_samples_de = sample(tm_de, n)
tm_plot = plotmatrix(unique_df_physical(tm_samples_br), unique_df_physical(tm_samples_de);
    dimlabels=labels, size=plot_size, xticklabelrotation=pi\2, legend_labels=data_labels,
    alphas=(0.8, 0.8, 0.8), ticklabelsize=9, markersize=2
)

laplace_samples_br = sample(laplace_br, n)
laplace_samples_de = sample(laplace_de, n)
laplace_plot = plotmatrix(unique_df_physical(laplace_samples_br), unique_df_physical(laplace_samples_de);
    dimlabels=labels, size=plot_size, xticklabelrotation=pi\2, legend_labels=data_labels,
    alphas=(0.8, 0.8, 0.8), ticklabelsize=9, markersize=2
)

## Plot Output
unique_output = unique(tmcmc_samples_de.f)
unique_output = permutedims(hcat(unique_output...))
output_tmcmc_de = plotmatrix(unique_output, data_de;
    dimlabels=[L"$f_1$ [Hz]", L"$f_2$ [Hz]", L"$f_3$ [Hz]", L"$f_4$ [Hz]"],
    size=output_plot_size,
    xticklabelrotation=pi\2,
    legend_labels=["Posterior", L"\biD^{(\mathrm{DE})}"], alphas=(0.8, 0.8, 0.8),
)

unique_output = unique(tmcmc_samples_br.f)
unique_output = permutedims(hcat(unique_output...))
output_tmcmc_br = plotmatrix(unique_output, data_br;
    dimlabels=[L"$f_1$ [Hz]", L"$f_2$ [Hz]", L"$f_3$ [Hz]", L"$f_4$ [Hz]"],
    size=output_plot_size,
    xticklabelrotation=pi\2,
    legend_labels=["Posterior", L"\biD^{(\mathrm{BR})}"], alphas=(0.8, 0.8, 0.8),
)

data_plot = plotmatrix(data_br, data_de;
    dimlabels=[L"$f_1$ [Hz]", L"$f_2$ [Hz]", L"$f_3$ [Hz]", L"$f_4$ [Hz]"],
    size=output_plot_size,
    xticklabelrotation=pi\2,
    alphas=(0.8, 0.8, 0.8),
    legend_labels =[L"\biD^{(\mathrm{BR})}", L"\biD^{(\mathrm{DE})}"])

## Plot Summary statistics
## Compute posterior mean and standard deviation
statistics_mh_de = get_mean_std(unique_df_physical(mh_samples_de))
statistics_mh_br = get_mean_std(unique_df_physical(mh_samples_br))

statistics_tmcmc_de = get_mean_std(unique_df_physical(tmcmc_samples_de))
statistics_tmcmc_br = get_mean_std(unique_df_physical(tmcmc_samples_br))

statistics_laplace_de = get_mean_std(unique_df_physical(laplace_samples_de))
statistics_laplace_br = get_mean_std(unique_df_physical(laplace_samples_br))

statistics_tm_de = get_mean_std(unique_df_physical(tm_samples_de))
statistics_tm_br = get_mean_std(unique_df_physical(tm_samples_br))

statistics_MLE_de = get_mean_std(MLEEstimate_physical_de)
statistics_MLE_br = get_mean_std(MLEEstimate_physical_br)

statistics_MAP_de = get_mean_std(MAPEstimate_physical_de)
statistics_MAP_br = get_mean_std(MAPEstimate_physical_br)

## Plot mean and standard deviation
results = [
    (name="M-H", de=statistics_mh_de, br=statistics_mh_br),
    (name="TMCMC", de=statistics_tmcmc_de, br=statistics_tmcmc_br),
    (name="Laplace", de=statistics_laplace_de, br=statistics_laplace_br),
    (name="TM", de=statistics_tm_de, br=statistics_tm_br),
    (name="MLE", de=statistics_MLE_de, br=statistics_MLE_br),
    (name="MAP", de=statistics_MAP_de, br=statistics_MAP_br),
]

summary_plot = plot_summary_statistics(results)

##
if save_results
    time_folder = joinpath(pwd(), "results", time_stamp)
    mkpath(time_folder)

    save(joinpath(time_folder, "data_comparison.png"), data_plot, px_per_unit=4)
    save(joinpath(time_folder, "post_output_de.png"), output_tmcmc_de, px_per_unit=4)
    save(joinpath(time_folder, "post_output_br.png"), output_tmcmc_br, px_per_unit=4)
    save(joinpath(time_folder, "tmcmc_samples.png"), tmcmc_plot, px_per_unit=4)
    save(joinpath(time_folder, "mh_samples.png"), mh_plot, px_per_unit=4)
    save(joinpath(time_folder, "laplace_samples.png"), laplace_plot, px_per_unit=4)
    save(joinpath(time_folder, "tm_samples.png"), tm_plot, px_per_unit=4)
    save(joinpath(time_folder, "summary_statistics.pdf"), summary_plot, px_per_unit=4)

    # save samples as CSV
    CSV.write(joinpath(time_folder,"mh_samples_de.csv"),  unique_df_physical(mh_samples_de))
    CSV.write(joinpath(time_folder,"mh_samples_br.csv"),  unique_df_physical(mh_samples_br))

    CSV.write(joinpath(time_folder,"tmcmc_samples_de.csv"),  unique_df_physical(tmcmc_samples_de))
    CSV.write(joinpath(time_folder,"tmcmc_samples_br.csv"),  unique_df_physical(tmcmc_samples_br))

    CSV.write(joinpath(time_folder,"laplace_samples_de.csv"),  unique_df_physical(laplace_samples_de))
    CSV.write(joinpath(time_folder,"laplace_samples_br.csv"),  unique_df_physical(laplace_samples_br))

    CSV.write(joinpath(time_folder,"tm_samples_de.csv"),  unique_df_physical(tm_samples_de))
    CSV.write(joinpath(time_folder,"tm_samples_br.csv"),  unique_df_physical(tm_samples_br))

    CSV.write(joinpath(time_folder,"mle_de.csv"),  MLEEstimate_physical_de)
    CSV.write(joinpath(time_folder,"mle_br.csv"),  MLEEstimate_physical_br)

    CSV.write(joinpath(time_folder,"map_de.csv"),  MAPEstimate_physical_de)
    CSV.write(joinpath(time_folder,"map_br.csv"),  MAPEstimate_physical_br)
end
