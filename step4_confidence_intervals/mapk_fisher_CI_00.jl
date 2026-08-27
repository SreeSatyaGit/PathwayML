#=
Step 4: Fisher Information Matrix (FIM) based confidence intervals for the mechanistic
parameters Step 3 classified as identifiable: kDuspStop, k43b1, kpCraf, kErkPhosPcraf,
kErkInbEgfr. CIs are NOT computed/reported for the other 55 parameters -- per the paper's
methodology, a CI for a non-identifiable parameter isn't meaningful.

ASSUMPTION: your real data doesn't include per-timepoint measurement uncertainty, so (matching
the assumption used elsewhere in this pipeline) we assume each measurement's uncertainty is
0.5% of that species' own min-max range. This is a real assumption, not derived from your
data -- adjust `measurement_uncertainty_fraction` below if you have a better estimate (e.g.
from replicate experiments or known assay precision).

TODO: NN architecture must match Step 2a's winner (trial 57: 2 layers, width 4), same as Step 3.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, DataFrames, CSV, Zygote

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

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
    Lux.Dense(2, 4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(4, 4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(4, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

all_results = deserialize("../step2b_model_trainer/res_mapk_hnode/mapk_hnode_00.jld")
successful = filter(r -> r.status == "success", all_results)
@assert !isempty(successful) "No successful Step 2b replicates found."
best = successful[argmin([r.final_training_cost for r in successful])]
println("Using best replicate, training cost = ", best.final_training_cost)

n_mech_params = length(free_idx_hnode_mapk)
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
tspan = (initial_time_training_mapk, end_time_training_mapk)
n_timepoints = length(timestamps_minutes_mapk)

##########################################################################################################################
##################################################### CONFIDENCE INTERVALS ##############################################

# Same explicit array-literal pattern as Step 3 -- avoids Zygote collapsing NamedTuple/Tuple
# iteration into a non-array output that jacobian() rejects.
extract_species(state) = [
    state[3], state[16] + state[18] + state[21], state[23], state[27], state[29], state[31],
    state[53], state[59], state[6], state[9], state[65], state[67],
]

function model(params, final_time)
    prob = ODEProblem{true}(hnode_derivative_function, params.u0, (0, final_time))
    sol = solve(prob, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=sensealg)
    return extract_species(sol.u[end])
end

first_point(p) = extract_species(p.u0)

# Reference magnitude per species, for scaling measurement uncertainty (same fitted-trajectory
# convention as Step 3)
function get_fitted_trajectory_maxima()
    prob = ODEProblem{true}(hnode_derivative_function, parameters_optimized_def.u0, tspan)
    sol = solve(prob, integrator, p=parameters_optimized_def.pars, saveat=timestamps_minutes_mapk, abstol=abstol, reltol=reltol, sensealg=sensealg)
    all_outputs = hcat([extract_species(sol.u[j]) for j in eachindex(sol.u)]...)
    return maximum(abs.(all_outputs), dims=2) .+ 1e-8
end

measurement_uncertainty_fraction = 0.005  # ASSUMPTION -- see header note
species_maxima = get_fitted_trajectory_maxima()

function get_covariance_matrix(parameters_to_consider)
    sensitivities = [Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1]]
    for j in 2:n_timepoints
        push!(sensitivities, Zygote.jacobian(p -> model(p, timestamps_minutes_mapk[j]), parameters_to_consider)[1])
    end
    sensitivity_matrix = vcat(sensitivities...)

    normalization_vec = vec(repeat(measurement_uncertainty_fraction .* species_maxima, n_timepoints))
    normalization_matrix = Diagonal(1 ./ normalization_vec)
    normalization_matrix = abs2.(normalization_matrix)

    observed_FIM = sensitivity_matrix' * normalization_matrix * sensitivity_matrix
    observed_FIM = Symmetric(observed_FIM)

    return pinv(observed_FIM)  # lower bound on the covariance matrix
end

println("Computing FIM (", 12 * n_timepoints, " Zygote.jacobian calls, may take a while)...")
cov = get_covariance_matrix(parameters_optimized_def)

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

# Load Step 3's identifiability classification -- only report CIs for these
identifiability_path = "../step3_parameters_identifiability/results/identifiability_summary_mapk_00.csv"
@assert isfile(identifiability_path) "Run Step 3 first -- $(identifiability_path) not found."
identifiability_df = CSV.read(identifiability_path, DataFrame)
identifiable_names = Set(identifiability_df[identifiability_df.identifiable .== true, :parameter])
println("Identifiable parameters from Step 3: ", identifiable_names)

results_ci = DataFrame(parameter=String[], estimate=Float64[], ci_lower=Float64[], ci_upper=Float64[])

for k in 1:n_mech_params
    name = mech_param_names[k]
    if !(name in identifiable_names)
        continue
    end
    idx = n_u0 + n_p_net + k
    estimate = parameters_optimized_def.pars.ode_par[k]
    variance = cov[idx, idx]
    if variance < 0
        println("WARNING: negative variance for ", name, " (", variance, ") -- pinv() numerical issue, skipping CI")
        continue
    end
    sd = sqrt(variance)
    ci_lower = estimate - 1.96 * sd
    ci_upper = estimate + 1.96 * sd
    push!(results_ci, (name, estimate, ci_lower, ci_upper))
end

println(results_ci)
CSV.write("results/mapk_fisher_CI_00.csv", results_ci)
println()
println("Full results: results/mapk_fisher_CI_00.csv")
