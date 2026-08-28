#=
Predicts pathway behavior under Vemurafenib + alpelisib (PI3K inhibitor), REPLACING
Trametinib, using the already-fitted Vem+Tram HNODE model (mechanistic parameters + trained
NN taken as-is, nothing refit). This is exploratory "what-if" simulation, not a validated
forecast -- no experimental data exists for this combination.

Produces three things, all exported to CSV (no plotting library involved):
  1. Vem+Tram baseline  -- re-simulates the ORIGINAL fitted condition, as a sanity check
     that this script's setup matches the validated Step 2b fit.
  2. Vem+PI3Ki central prediction -- Trametinib off, alpelisib on at clinical Cmax.
  3. Vem+PI3Ki uncertainty band -- since k_PIP2_to_PIP3, k_PI3K_recruit, kMTOR_Feedback,
     and k_PTEN were classified NON-IDENTIFIABLE by Step 3 (the Vem+Tram data barely
     constrains this submodule), the central prediction alone would overstate confidence.
     This perturbs those specific parameters across a plausible range (+/-3x, matching the
     rough scale of the parameter search bounds used elsewhere in this pipeline) and reports
     the resulting spread, not a single line.

TODO: NN architecture must match Step 2a's winner (trial 57: 2 layers, width 4).
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, Random, DataFrames, CSV, Statistics

if !isdir("results")
    mkdir("results")
end

include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")       # defines assemble_full_parameter_vector, and the validated get_uode_model_function
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3ki_extension_functions.jl")           # defines get_uode_model_function_with_pi3ki
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_settings.jl")

integrator = TRBDF2(autodiff=false)
abstol = 1e-8
reltol = 1e-6

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
fitted_pars = best.parameters_training   # ComponentArray: .p_net (trained NN), .ode_par (absolute fitted values)

tspan = (initial_time_training_mapk, end_time_training_mapk)
tfine_minutes = range(tspan[1], tspan[2], length=200)

# ---- alpelisib parameterization (see mapk_pi3ki_extension_functions.jl header for sources) ----
alpelisib_ic50_M = 4.6e-9        # biochemical IC50, PI3Kalpha, molar
alpelisib_hill_n = 1.0           # assumed standard competitive inhibition (not separately reported)
alpelisib_clinical_conc_M = 5.62e-6   # ~2480 ng/mL steady-state Cmax at approved 300mg QD dose, converted using MW=441.47 g/mol

##########################################################################################################################
################################################## SCENARIO 1: Vem+Tram baseline #########################################

println("Simulating Scenario 1: Vem+Tram (fitted baseline, sanity check)...")
hnode_fn_baseline = get_uode_model_function(approximating_neural_network, best.net_status, ones(n_mech_params), free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_mapk)
prob_baseline = ODEProblem{true}(hnode_fn_baseline, original_u0_mapk, tspan)
sol_baseline = solve(prob_baseline, integrator, p=fitted_pars, saveat=tfine_minutes, abstol=abstol, reltol=reltol)
@assert sol_baseline.retcode == SciMLBase.ReturnCode.Success "Baseline re-simulation failed -- check fitted_pars/architecture match Step 2b."

##########################################################################################################################
############################################# SCENARIO 2: Vem+PI3Ki (Tram replaced) ######################################

println("Simulating Scenario 2: Vem+PI3Ki (central prediction, Trametinib replaced by alpelisib)...")

# fixed_values_mapk order matches fixed_idx_mapk = [54,55,56,57,58,62,63]:
# [Tram.conc, Ki_RAF, Ki_KSR, Tram.Hill_n, Vem.conc, Vem.IC50, Vem.Hill_n]
fixed_values_no_tram = copy(fixed_values_mapk)
fixed_values_no_tram[1] = 0.0   # Tram.conc = 0 -- Trametinib absent

hnode_fn_pi3ki = get_uode_model_function_with_pi3ki(
    approximating_neural_network, best.net_status, ones(n_mech_params),
    free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_no_tram,
    alpelisib_clinical_conc_M, alpelisib_ic50_M, alpelisib_hill_n
)
prob_pi3ki = ODEProblem{true}(hnode_fn_pi3ki, original_u0_mapk, tspan)
sol_pi3ki_central = solve(prob_pi3ki, integrator, p=fitted_pars, saveat=tfine_minutes, abstol=abstol, reltol=reltol)
@assert sol_pi3ki_central.retcode == SciMLBase.ReturnCode.Success "Vem+PI3Ki central prediction failed to integrate."

##########################################################################################################################
######################################### SCENARIO 3: uncertainty band (non-identifiable params) #########################

println("Simulating uncertainty band (perturbing non-identifiable PI3K-submodule parameters)...")

# indices of these parameters WITHIN free_idx_hnode_mapk / fitted_pars.ode_par
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
uncertain_params = ["k_PIP2_to_PIP3", "k_PI3K_recruit", "kMTOR_Feedback", "k_PTEN"]
uncertain_indices = [findfirst(==(name), mech_param_names) for name in uncertain_params]
@assert all(!isnothing, uncertain_indices) "Could not find all uncertain parameter names -- check spelling against mech_param_names."

n_samples = 20
perturbation_range = (1/3, 3.0)   # multiplicative range, matching the rough scale of bounds used elsewhere
rng = Random.default_rng()
Random.seed!(rng, 42)

species_names_grid = ["pEGFR", "panRAS", "pCRAF", "pMEK", "pERK", "DUSP", "pAKT", "p4EBP1", "pHer2", "pHer3", "pDGFR", "pS6K"]
n_fine = length(tfine_minutes)
envelope = Dict{String, Matrix{Float64}}()
for name in species_names_grid
    envelope[name] = zeros(n_samples, n_fine)
end

for s in 1:n_samples
    perturbed_ode_par = copy(fitted_pars.ode_par)
    for idx in uncertain_indices
        factor = exp(rand(rng) * (log(perturbation_range[2]) - log(perturbation_range[1])) + log(perturbation_range[1]))
        perturbed_ode_par[idx] *= factor
    end
    perturbed_pars = ComponentArray(p_net=fitted_pars.p_net, ode_par=perturbed_ode_par)

    local sol
    try
        sol = solve(prob_pi3ki, integrator, p=perturbed_pars, saveat=tfine_minutes, abstol=abstol, reltol=reltol)
        if sol.retcode != SciMLBase.ReturnCode.Success
            continue
        end
    catch
        continue
    end

    for (name, extractor) in pairs(OBSERVED_SPECIES)
        envelope[String(name)][s, :] = [extractor(sol.u[j]) for j in eachindex(sol.u)]
    end
end

##########################################################################################################################
######################################################### EXPORT #########################################################

df = DataFrame(t_hours = collect(tfine_minutes) ./ 60.0)

for (name, extractor) in pairs(OBSERVED_SPECIES)
    baseline_raw = [extractor(sol_baseline.u[j]) for j in eachindex(sol_baseline.u)]
    central_raw = [extractor(sol_pi3ki_central.u[j]) for j in eachindex(sol_pi3ki_central.u)]

    df[!, "$(name)_vemtram_baseline_norm"] = min_max_normalize(baseline_raw)
    df[!, "$(name)_vempi3ki_central_norm"] = min_max_normalize(central_raw)

    band_min = vec(minimum(envelope[String(name)], dims=1))
    band_max = vec(maximum(envelope[String(name)], dims=1))
    band_median = vec(median(envelope[String(name)], dims=1))
    df[!, "$(name)_vempi3ki_band_min_norm"] = min_max_normalize(band_min)
    df[!, "$(name)_vempi3ki_band_max_norm"] = min_max_normalize(band_max)
    df[!, "$(name)_vempi3ki_band_median_norm"] = min_max_normalize(band_median)
end

CSV.write("results/vem_pi3ki_prediction.csv", df)
println()
println("Wrote results/vem_pi3ki_prediction.csv (", nrow(df), " timepoints x ", ncol(df), " columns)")
println("Columns per species: baseline (fitted Vem+Tram), central prediction (Vem+PI3Ki),")
println("and a min/max/median uncertainty band from perturbing non-identifiable PI3K parameters.")