#=
Step 3: local identifiability-at-a-point analysis for the fitted Vem+PI3K HNODE model.
Same Hessian null-space methodology as the Vem+Tram Step 3, adapted for this condition's
61 free parameters (60 shared mechanistic + 1 fitted PI3K IC50) and 9 observed species.

TODO: NN architecture must match Step 2a's winner for THIS condition -- confirmed as 2
hidden layers, width 8 from the trained net_status structure.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, Random, DataFrames, CSV, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Zygote

epsilon = 1e-5
delta = 0.05

if !isdir("results")
    mkdir("results")
end

include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")   # assemble_full_parameter_vector
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_settings.jl")     # base params/ICs/fixed idx
include("../test_case_settings/mapk_pi3k_vempi3ki_settings/mapk_pi3k_vempi3ki_hnode_functions.jl")  # get_uode_model_function_vempi3ki
include("../test_case_settings/mapk_pi3k_vempi3ki_settings/mapk_pi3k_vempi3ki_model_settings.jl")   # this condition's data/species/weights

integrator = TRBDF2(autodiff=false)
abstol = 1e-8
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

# TODO: must match Step 2a's winner for THIS condition -- confirmed as 2 hidden layers,
# width 8 from the trained net_status structure (layer_1, layer_2, layer_3)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
    Lux.Dense(2, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

all_results = deserialize("../step2b_model_trainer/res_vempi3ki_hnode/vempi3ki_hnode_00.jld")
successful = filter(r -> r.status == "success", all_results)
@assert !isempty(successful) "No successful Step 2b replicates found."
best = successful[argmin([r.final_training_cost for r in successful])]
println("Using best replicate, training cost = ", best.final_training_cost)

n_mech_params = length(free_idx_hnode_mapk)   # 60
n_total_params = n_mech_params + 1            # + fitted PI3K IC50
tspan = (initial_time_training_vempi3ki, end_time_training_vempi3ki)

fixed_values_vempi3ki = copy(fixed_values_mapk)
fixed_values_vempi3ki[1] = 0.0   # Trametinib off

# .ode_par in the saved result is already ABSOLUTE (mechanistic params scaled by their
# pretrained values, PI3K IC50 already converted from its scale factor) -- reconstruct with
# scale = 1 to avoid double-scaling, same convention as Vem+Tram's Step 3.
hnode_derivative_function = get_uode_model_function_vempi3ki(
    approximating_neural_network, best.net_status, ones(n_mech_params),
    free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_vempi3ki,
    pi3ki_conc_ref, pi3ki_hill_n_ref, 1.0   # nominal=1.0 so ode_par's last slot IS the absolute IC50 directly
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

# Written out explicitly (array literal, no closures/iteration) for this condition's 9
# species -- same Zygote-safety pattern as Vem+Tram's Step 3 (avoids NamedTuple/Tuple
# collapse inside differentiated code).
# Order: pEGFR, pCRAF, pMEK, pERK, DUSP, pAKT, IGF1R, pHer2, pHer3
extract_species_vempi3ki(state) = [
    state[3], state[23], state[27], state[29], state[31], state[53], state[40], state[6], state[9],
]
const species_names_ordered_vempi3ki = ["pEGFR", "pCRAF", "pMEK", "pERK", "DUSP", "pAKT", "IGF1R", "pHer2", "pHer3"]

function model(params, final_time)
    prob = ODEProblem{true}(hnode_derivative_function, params.u0, (0, final_time))
    sol = solve(prob, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=sensealg)
    return extract_species_vempi3ki(sol.u[end])
end

first_point(p) = extract_species_vempi3ki(p.u0)

n_timepoints = length(timestamps_minutes_vempi3ki)
n_species = length(species_names_ordered_vempi3ki)

function get_fitted_trajectory_maxima()
    prob = ODEProblem{true}(hnode_derivative_function, parameters_optimized_def.u0, tspan)
    sol = solve(prob, integrator, p=parameters_optimized_def.pars, saveat=timestamps_minutes_vempi3ki, abstol=abstol, reltol=reltol, sensealg=sensealg)
    all_outputs = hcat([extract_species_vempi3ki(sol.u[j]) for j in eachindex(sol.u)]...)
    return maximum(abs.(all_outputs), dims=2) .+ 1e-8
end

gamma = get_fitted_trajectory_maxima()

function get_Hessian_Spectrum(parameters_to_consider)
    sensitivities = [Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1] .* parameters_to_consider']
    for j in 2:n_timepoints
        push!(sensitivities, Zygote.jacobian(p -> model(p, timestamps_minutes_vempi3ki[j]), parameters_to_consider)[1] .* parameters_to_consider')
    end
    sensitivity_matrix = vcat(sensitivities...)

    normalization_vec = vec(repeat(gamma, n_timepoints))
    normalization_matrix = Diagonal(1 ./ normalization_vec)
    normalization_matrix = abs2.(normalization_matrix)

    hessian = sensitivity_matrix' * normalization_matrix * sensitivity_matrix .* 1 / (n_species * n_timepoints)
    hessian = Symmetric(hessian)
    eig = eigen(hessian)

    eigen_values = real.(eig.values)
    eigen_vectors = real.(eig.vectors)'
    return hcat(eigen_vectors, eigen_values)
end

println("Computing sensitivities and Hessian (", n_species * n_timepoints, " Zygote.jacobian calls, may take a while)...")
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
mech_param_names = vcat([all_param_names[i] for i in free_idx_hnode_mapk], ["PI3Ki_IC50"])
@assert length(mech_param_names) == n_total_params

results_summary = DataFrame(parameter=String[], mech_component=Float64[], nn_component=Float64[], identifiable=Bool[])

for k in 1:n_total_params
    par_index_from_end = n_total_params - k
    projection = get_projection_on_null_space(null_direction_dataframe, par_index_from_end)

    mech_component = sum(abs2, projection[end-n_total_params+1:end])
    nn_component = sum(abs2, projection[n_u0+1:end-n_total_params])
    total_norm = mech_component + nn_component + sum(abs2, projection[1:n_u0])

    is_identifiable = total_norm < delta
    push!(results_summary, (mech_param_names[k], mech_component, nn_component, is_identifiable))
end

sort!(results_summary, :mech_component, rev=true)
println(results_summary)
CSV.write("results/identifiability_summary_vempi3ki_00.csv", results_summary)

n_identifiable = count(results_summary.identifiable)
println()
println(n_identifiable, " / ", n_total_params, " parameters classified as identifiable.")
println("Full results: results/identifiability_summary_vempi3ki_00.csv")