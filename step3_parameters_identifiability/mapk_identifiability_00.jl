#=
Step 3: local identifiability-at-a-point analysis for the fitted real MAPK/PI3K + Vem/Tram
HNODE model. Loads the best replicate from Step 2b, computes the Gauss-Newton Hessian of
the trajectory-sensitivity function at the 6 real timepoints across all 12 observed
species, finds its near-null eigenspace, and reports each of the 60 mechanistic
parameters' projection onto that subspace (split into mechanistic vs. NN contribution).

TODO: this NN architecture MUST match whatever Step 2a selected as best (currently
hardcoded to trial 57's 2-layer, width-4 architecture -- update if you re-run Step 2a
and get a different winner).
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, Random, DataFrames, CSV, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Zygote

epsilon = 1e-5   # threshold distinguishing zero eigenvalues (paper default)
delta = 0.05     # threshold for classifying a parameter as non-identifiable (paper default)

if !isdir("results")
    mkdir("results")
end

include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_settings.jl")

integrator = TRBDF2(autodiff=false)
abstol = 1e-8
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

# TODO: must match Step 2a's winning architecture exactly (trial 57: 2 layers, width 4)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
    Lux.Dense(2, 4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(4, 4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(4, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

# Load Step 2b results, pick the best successful replicate
all_results = deserialize("../step2b_model_trainer/res_mapk_hnode/mapk_hnode_00.jld")
successful = filter(r -> r.status == "success", all_results)
@assert !isempty(successful) "No successful Step 2b replicates found."
best = successful[argmin([r.final_training_cost for r in successful])]
println("Using best replicate, training cost = ", best.final_training_cost)

n_mech_params = length(free_idx_hnode_mapk)   # 60
tspan = (initial_time_training_mapk, end_time_training_mapk)

# .ode_par in the saved result is already the ABSOLUTE fitted value (not a scale factor --
# Step 2b multiplies by pretrained_ode_pars before saving), so reconstruct the derivative
# function with original_parameters_opt = ones(...) to avoid double-scaling.
hnode_derivative_function = get_uode_model_function(
    approximating_neural_network, best.net_status, ones(n_mech_params),
    free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_mapk
)

parameters_optimized = ComponentArray(best.parameters_training)
u0 = ComponentArray(original_u0_mapk)
parameters_optimized_def = ComponentArray{eltype(parameters_optimized.p_net)}()
parameters_optimized_def = ComponentArray(parameters_optimized_def; u0)
pars = ComponentArray(parameters_optimized)
parameters_optimized_def = ComponentArray(parameters_optimized_def; pars)

n_u0 = length(u0)
n_p_net = length(parameters_optimized.p_net)

##########################################################################################################################
##################################################### IDENTIFIABILITY ###################################################

# Model output: the 12 observed species (raw, not normalized) at a given time
function model(params, final_time)
    prob = ODEProblem{true}(hnode_derivative_function, params.u0, (0, final_time))
    sol = solve(prob, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=sensealg)
    state = sol.u[end]
    return [extractor(state) for extractor in OBSERVED_SPECIES]
end

first_point(p) = [extractor(p.u0) for extractor in OBSERVED_SPECIES]

n_timepoints = length(timestamps_minutes_mapk)

# Normalization factor per observed species, based on the fitted model's own trajectory
# magnitude across the 6 timepoints (matches the paper's gamma_i convention)
function get_fitted_trajectory_maxima()
    prob = ODEProblem{true}(hnode_derivative_function, parameters_optimized_def.u0, tspan)
    sol = solve(prob, integrator, p=parameters_optimized_def.pars, saveat=timestamps_minutes_mapk, abstol=abstol, reltol=reltol, sensealg=sensealg)
    all_outputs = hcat([[extractor(sol.u[j]) for extractor in OBSERVED_SPECIES] for j in eachindex(sol.u)]...)
    return maximum(abs.(all_outputs), dims=2) .+ 1e-8  # avoid div-by-zero for a flat species
end

gamma = get_fitted_trajectory_maxima()  # 12-element vector

function get_Hessian_Spectrum(parameters_to_consider)
    sensitivities = [Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1] .* parameters_to_consider']
    for j in 2:n_timepoints
        push!(sensitivities, Zygote.jacobian(p -> model(p, timestamps_minutes_mapk[j]), parameters_to_consider)[1] .* parameters_to_consider')
    end
    sensitivity_matrix = vcat(sensitivities...)  # (12*6) x n_params

    normalization_vec = vec(repeat(gamma, n_timepoints))
    normalization_matrix = Diagonal(1 ./ normalization_vec)
    normalization_matrix = abs2.(normalization_matrix)

    hessian = sensitivity_matrix' * normalization_matrix * sensitivity_matrix .* 1 / (12 * n_timepoints)
    hessian = Symmetric(hessian)
    eig = eigen(hessian)

    eigen_values = real.(eig.values)
    eigen_vectors = real.(eig.vectors)'
    return hcat(eigen_vectors, eigen_values)
end

println("Computing sensitivities and Hessian (this involves ", 12 * n_timepoints, " Zygote.jacobian calls, may take a while)...")
eigen_vectors_with_eigen_values = get_Hessian_Spectrum(parameters_optimized_def)
null_direction_dataframe = eigen_vectors_with_eigen_values[abs.(eigen_vectors_with_eigen_values[:, end]) .< epsilon, :]
println("Null-space dimension: ", size(null_direction_dataframe, 1), " / ", size(eigen_vectors_with_eigen_values, 1))

function get_projection_on_null_space(null_direction_dataframe, par_index_from_end)
    parameter_versor = zeros(size(parameters_optimized_def))
    parameter_versor[end-par_index_from_end] = 1

    projection = zeros(size(parameters_optimized_def))
    for i in 1:size(null_direction_dataframe)[1]
        projection += dot(parameter_versor, null_direction_dataframe[i, 1:end-1]') .* null_direction_dataframe[i, 1:end-1]
    end
    return projection
end

# Human-readable names for the 60 free mechanistic parameters, in free_idx_hnode_mapk order
const all_param_names = [
    "ka1","kr1","kc1","kpCraf","kpMek","kpErk","kDegradEgfr","kErkInbEgfr","kShcDephos","kptpDeg",
    "kGrb2CombShc","kSprtyInbGrb2","kSosCombGrb2","kErkPhosSos","kErkPhosPcraf","kPcrafDegrad","kErkPhosMek","kMekDegrad",
    "kDuspInbErk","kErkDeg","kinbBraf","kDuspStop","kDusps","kSproutyForm","kSprtyComeDown","kdegrad",
    "km_Sprty_decay","km_Dusp","km_Sprty","kErkDephos","kDuspDeg","kHer2_act","kHer3_act","k_p85_bind_EGFR",
    "k_p85_bind_Her2","k_p85_bind_Her3","k_p85_bind_IGFR","k_p85_unbind","k_PI3K_recruit","kMTOR_Feedback",
    "k_PIP2_to_PIP3","k_PTEN","kAkt","kdegradAKT","kb1","k43b1","k4ebp1","k_4EBP1_dephos","kKSRphos","kKSRdephos",
    "kMekByBraf","kMekByCraf","kMekByKSR","Tram_conc","K_tram_RAF","K_tram_KSR","Tram_Hill_n","Vem_conc",
    "kDimerForm","kDimerDissoc","kParadoxCRAF","Vem_IC50","Vem_Hill_n","kPDGFR_act","k_p85_bind_PDGFR",
    "kS6K_phos","kS6K_dephos","K_displace",
]
mech_param_names = [all_param_names[i] for i in free_idx_hnode_mapk]
@assert length(mech_param_names) == n_mech_params

results_summary = DataFrame(parameter=String[], mech_component=Float64[], nn_component=Float64[], identifiable=Bool[])

for k in 1:n_mech_params
    par_index_from_end = n_mech_params - k
    projection = get_projection_on_null_space(null_direction_dataframe, par_index_from_end)

    mech_component = sum(abs2, projection[end-n_mech_params+1:end])
    nn_component = sum(abs2, projection[n_u0+1:end-n_mech_params])
    total_norm = mech_component + nn_component + sum(abs2, projection[1:n_u0])

    is_identifiable = total_norm < delta
    push!(results_summary, (mech_param_names[k], mech_component, nn_component, is_identifiable))
end

sort!(results_summary, :mech_component, rev=true)
println(results_summary)
CSV.write("results/identifiability_summary_mapk_00.csv", results_summary)

n_identifiable = count(results_summary.identifiable)
println()
println(n_identifiable, " / ", n_mech_params, " mechanistic parameters classified as identifiable.")
println("Full results: results/identifiability_summary_mapk_00.csv")
