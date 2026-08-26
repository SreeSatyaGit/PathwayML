#=
Step 4: Fisher Information Matrix (FIM) based confidence intervals for the mechanistic
parameters classified as identifiable in Step 3.

TODO: only trust the CIs for parameters Step 3 marked identifiable=true -- the FIM-based
CI for a non-identifiable parameter is meaningless (often absurdly narrow or wide).
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux, Zygote, StatsPlots, LaTeXStrings

error_level = "e0.0"

if !isdir("results")
    mkdir("results")
end

include("../test_case_settings/vem_tram_model_settings/vem_tram_model_functions.jl")
include("../test_case_settings/vem_tram_model_settings/vem_tram_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5"]

integrator = TRBDF2(autodiff=false)
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
# TODO: must match the architecture used in Step 2b / Step 3 exactly
approximating_neural_network = Lux.Chain(
    Lux.Dense(3, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform)
)

par_opt = deserialize("../step2b_model_trainer/res_vem_tram/vem_tram_00.jld")[1]  # TODO: pick your best replicate

ode_data = deserialize("../datasets/e0.0/data/ode_data_vem_tram.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_vem_tram.jld")

rng = Random.default_rng()
Random.seed!(rng, 0)

tspan = (initial_time_training, end_time_training)
tsteps = solution_dataframe.t
n_timepoints = length(tsteps)
n_mech_params = length(original_parameters)

uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, ones(n_mech_params))

parameters_optimized_def = ComponentArray{eltype(par_opt.parameters_training.p_net)}()
u0 = ComponentArray(par_opt.initial_state_training[:, 1])
parameters_optimized_def = ComponentArray(parameters_optimized_def; u0)
pars = ComponentArray(par_opt.parameters_training)
parameters_optimized_def = ComponentArray(parameters_optimized_def; pars)

n_u0 = length(u0)

##########################################################################################################################
##################################################### CONFIDENCE INTERVALS ##############################################

function model(params, final_time)
    prob = ODEProblem{true}(uode_derivative_function, params.u0, (0, final_time))
    solutions = solve(prob, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=sensealg)
    return Array(solutions)[:, end]
end

first_point(p) = p.u0

function get_covariance_matrix(parameters_to_consider; measurement_uncertainty_fraction=0.005)
    sensitivities = [Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1]]
    for j in 2:n_timepoints
        push!(sensitivities, Zygote.jacobian(p -> model(p, tsteps[j]), parameters_to_consider)[1])
    end
    sensitivity_matrix = vcat(sensitivities...)

    normalization_factor = maximum(ode_data, dims=2)
    normalization_matrix = vec(repeat(normalization_factor, n_timepoints))
    normalization_matrix = Diagonal(1 ./ (measurement_uncertainty_fraction .* normalization_matrix))
    normalization_matrix = abs2.(normalization_matrix)

    observed_FIM = sensitivity_matrix' * normalization_matrix * sensitivity_matrix
    observed_FIM = Symmetric(observed_FIM)

    return pinv(observed_FIM)  # lower bound on the covariance matrix
end

cov = get_covariance_matrix(parameters_optimized_def)

mech_param_names = ["k_raf", "d_raf", "k_mek", "d_mek", "k_erk", "d_erk", "k_pi3k", "d_pi3k", "k_akt", "d_akt"]
n_p_net = length(par_opt.parameters_training.p_net)

results_ci = DataFrame(parameter=String[], estimate=Float64[], ci_lower=Float64[], ci_upper=Float64[])

for k in 1:n_mech_params
    idx = n_u0 + n_p_net + k  # position of mechanistic parameter k in the flattened vector
    estimate = parameters_optimized_def.pars.ode_par[k]
    sd = sqrt(cov[idx, idx])
    ci_lower = estimate - 1.96 * sd
    ci_upper = estimate + 1.96 * sd
    push!(results_ci, (mech_param_names[k], estimate, ci_lower, ci_upper))
end

println(results_ci)
CSV.write("results/vem_tram_fisher_CI_00.csv", results_ci)
Serialization.serialize("results/vem_tram_opt_00_fisher_CI.jld", results_ci)
