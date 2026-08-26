#=
Step 3: local identifiability-at-a-point analysis for the vem_tram HNODE model (DS_0.0).
Computes the Gauss-Newton Hessian of the trajectory-sensitivity function, finds its
near-null eigenspace (threshold epsilon), and reports each mechanistic parameter's
squared-norm projection onto that subspace, split into "mechanistic" vs "NN" components
(threshold delta decides identifiable vs non-identifiable).

TODO: point `par_opt` at wherever Step 2b's `res_vem_tram/vem_tram_00.jld` results are saved,
and pick which trained replicate (lowest validation cost) to analyze.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux, Zygote, StatsPlots, LaTeXStrings

error_level = "e0.0"
epsilon = 1e-5   # threshold distinguishing zero eigenvalues (paper default)
delta = 0.05     # threshold for classifying a parameter as non-identifiable (paper default)

if !isdir("plots")
    mkdir("plots")
end

include("../test_case_settings/vem_tram_model_settings/vem_tram_model_functions.jl")
include("../test_case_settings/vem_tram_model_settings/vem_tram_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5"]

integrator = TRBDF2(autodiff=false)
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
# TODO: this NN architecture MUST match the one selected in Step 2a/2b exactly
approximating_neural_network = Lux.Chain(
    Lux.Dense(3, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform)
)

# TODO: point this at your best trained replicate from Step 2b
par_opt = deserialize("../step2b_model_trainer/res_vem_tram/vem_tram_00.jld")[1]  # e.g. pick lowest validation cost manually

ode_data = deserialize("../datasets/e0.0/data/ode_data_vem_tram.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_vem_tram.jld")

rng = Random.default_rng()
Random.seed!(rng, 0)

tspan = (initial_time_training, end_time_training)
tsteps = solution_dataframe.t
n_timepoints = length(tsteps)
n_mech_params = length(original_parameters)

uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, ones(n_mech_params))

parameters_optimized = ComponentArray(par_opt.parameters_training)
parameters_optimized_def = ComponentArray{eltype(parameters_optimized.p_net)}()
u0 = ComponentArray(par_opt.initial_state_training[:, 1])
parameters_optimized_def = ComponentArray(parameters_optimized_def; u0)
pars = ComponentArray(parameters_optimized)
parameters_optimized_def = ComponentArray(parameters_optimized_def; pars)

n_u0 = length(u0)
n_p_net = length(parameters_optimized.p_net)

##########################################################################################################################
##################################################### IDENTIFIABILITY ###################################################

function model(params, final_time)
    prob = ODEProblem{true}(uode_derivative_function, params.u0, (0, final_time))
    solutions = solve(prob, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=sensealg)
    return Array(solutions)[:, end]
end

first_point(p) = p.u0

function get_Hessian_Spectrum(parameters_to_consider)
    # sensitivity of the trajectory at each observed time point w.r.t. every parameter
    sensitivities = [Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1] .* parameters_to_consider']
    for j in 2:n_timepoints
        push!(sensitivities, Zygote.jacobian(p -> model(p, tsteps[j]), parameters_to_consider)[1] .* parameters_to_consider')
    end
    sensitivity_matrix = vcat(sensitivities...)

    normalization_factor = maximum(ode_data, dims=2)
    normalization_matrix = vec(repeat(normalization_factor, n_timepoints))
    normalization_matrix = Diagonal(1 ./ normalization_matrix)
    normalization_matrix = abs2.(normalization_matrix)

    hessian = sensitivity_matrix' * normalization_matrix * sensitivity_matrix .* 1 / size(solution_dataframe, 1) .* 1 / (size(solution_dataframe, 2) - 1)
    hessian = Symmetric(hessian)
    eig = eigen(hessian)

    eigen_values = real.(eig.values)
    eigen_vectors = real.(eig.vectors)'
    return hcat(eigen_vectors, eigen_values)
end

eigen_vectors_with_eigen_values = get_Hessian_Spectrum(parameters_optimized_def)
null_direction_dataframe = eigen_vectors_with_eigen_values[abs.(eigen_vectors_with_eigen_values[:, end]) .< epsilon, :]

function get_projection_on_null_space(null_direction_dataframe, par_index_from_end)
    parameter_versor = zeros(size(parameters_optimized_def))
    parameter_versor[end-par_index_from_end] = 1

    projection = zeros(size(parameters_optimized_def))
    for i in 1:size(null_direction_dataframe)[1]
        projection += dot(parameter_versor, null_direction_dataframe[i, 1:end-1]') .* null_direction_dataframe[i, 1:end-1]
    end
    return projection
end

# indices in parameters_optimized_def: [u0 (n_u0) | p_net (n_p_net) | ode_par (n_mech_params)]
# so mechanistic parameter k (1-indexed) sits at position (end - n_mech_params + k)
mech_param_names = ["k_raf", "d_raf", "k_mek", "d_mek", "k_erk", "d_erk", "k_pi3k", "d_pi3k", "k_akt", "d_akt"]

results_summary = DataFrame(parameter=String[], mech_component=Float64[], nn_component=Float64[], identifiable=Bool[])

for k in 1:n_mech_params
    par_index_from_end = n_mech_params - k  # 0-indexed offset from the end
    projection = get_projection_on_null_space(null_direction_dataframe, par_index_from_end)

    mech_component = sum(abs2, projection[end-n_mech_params+1:end])
    nn_component = sum(abs2, projection[n_u0+1:end-n_mech_params])
    total_norm = mech_component + nn_component + sum(abs2, projection[1:n_u0])

    is_identifiable = total_norm < delta
    push!(results_summary, (mech_param_names[k], mech_component, nn_component, is_identifiable))
end

println(results_summary)
CSV.write("plots/identifiability_summary_vem_tram_00.csv", results_summary)

# stacked bar plot: mechanistic vs NN contribution to each parameter's null-space projection
gr()
plt = Plots.plot(xtickfontsize=10, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfontsize=12, dpi=300, size=(700, 500))
groupedbar!(plt, hcat(results_summary.mech_component, results_summary.nn_component),
    bar_position=:stack, bar_width=0.6,
    xticks=(1:n_mech_params, mech_param_names), xrotation=45,
    label=["mechanistic" "NN"])
hline!(plt, [delta], color="red", linestyle=:dot, linewidth=2, label="threshold (delta)")
yaxis!(plt, "Squared norm of null-space projection")
Plots.savefig(plt, "plots/composition_projections_vem_tram_e00.svg")
