#=
Dose-response exploration using the fitted Vem+Tram HNODE model (mechanistic parameters +
trained NN taken as-is, nothing refit). Unlike the Vem+PI3Ki exercise, this stays within
the SAME two drugs and mechanisms the model was actually fit against -- but the two sweeps
below carry different extrapolation risk, and are kept deliberately separate for that reason:

  SWEEP 1 (Trametinib dose, Vem held at its fitted value): Trametinib's effect is pure
  mechanistic math (Ki_effective / f_MEK_activity, real Hill/displacement kinetics with
  fitted parameters) -- no neural network involved. Varying its dose is a straightforward
  interpolation/extrapolation of an equation that was actually fit. LOWER extrapolation risk.

  SWEEP 2 (Vemurafenib dose, Tram held at its fitted value): Vem.conc was FIXED (not
  varied) during training -- meaning the neural network's second input (Vemurafenib) was
  ALSO constant throughout training. The NN never learned how paradox-activation actually
  depends on Vemurafenib concentration; it only ever saw one fixed value. Varying Vem dose
  therefore also pushes the NN into genuinely untested input territory, in addition to the
  (real, fitted) mechanistic dose-response in kBRAF_eff. HIGHER extrapolation risk -- treat
  this sweep's results with more caution than Sweep 1's.

Fitted/training dose values (for reference): Tram.conc = 1e-6, Vem.conc = 1.0
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, DataFrames, CSV

if !isdir("results")
    mkdir("results")
end

include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")
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
fitted_pars = best.parameters_training

tspan = (initial_time_training_mapk, end_time_training_mapk)
tfine_minutes = range(tspan[1], tspan[2], length=200)

# fixed_values_mapk order matches fixed_idx_mapk = [54,55,56,57,58,62,63]:
# [Tram.conc, Ki_RAF, Ki_KSR, Tram.Hill_n, Vem.conc, Vem.IC50, Vem.Hill_n]
const TRAM_IDX = 1
const VEM_IDX = 5
const tram_fitted = fixed_values_mapk[TRAM_IDX]   # 1e-6
const vem_fitted = fixed_values_mapk[VEM_IDX]     # 1.0

function simulate_dose(tram_conc, vem_conc)
    fv = copy(fixed_values_mapk)
    fv[TRAM_IDX] = tram_conc
    fv[VEM_IDX] = vem_conc
    hnode_fn = get_uode_model_function(approximating_neural_network, best.net_status, ones(n_mech_params), free_idx_hnode_mapk, fixed_idx_mapk, fv)
    prob = ODEProblem{true}(hnode_fn, original_u0_mapk, tspan)
    sol = solve(prob, integrator, p=fitted_pars, saveat=tfine_minutes, abstol=abstol, reltol=reltol)
    return sol
end

##########################################################################################################################
############################################### SWEEP 1: Trametinib dose #################################################

println("Sweep 1: Trametinib dose-response (Vemurafenib held at fitted value)...")
tram_multipliers = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]   # 0 = Trametinib absent; 1.0 = fitted/training dose

df_tram = DataFrame(t_hours = collect(tfine_minutes) ./ 60.0)
for mult in tram_multipliers
    sol = simulate_dose(tram_fitted * mult, vem_fitted)
    if sol.retcode != SciMLBase.ReturnCode.Success
        println("  WARNING: Tram multiplier ", mult, "x failed to integrate (retcode=", sol.retcode, "), skipping")
        continue
    end
    label = "tram_$(mult)x"
    for (name, extractor) in pairs(OBSERVED_SPECIES)
        raw = [extractor(sol.u[j]) for j in eachindex(sol.u)]
        df_tram[!, "$(name)_$(label)_norm"] = min_max_normalize(raw)
    end
end
CSV.write("results/tram_dose_response.csv", df_tram)
println("  Wrote results/tram_dose_response.csv")

##########################################################################################################################
############################################### SWEEP 2: Vemurafenib dose ################################################

println("Sweep 2: Vemurafenib dose-response (Trametinib held at fitted value) -- HIGHER extrapolation risk, see header...")
vem_multipliers = [0.1, 0.3, 1.0, 3.0, 10.0]   # 1.0 = fitted/training dose

df_vem = DataFrame(t_hours = collect(tfine_minutes) ./ 60.0)
for mult in vem_multipliers
    sol = simulate_dose(tram_fitted, vem_fitted * mult)
    if sol.retcode != SciMLBase.ReturnCode.Success
        println("  WARNING: Vem multiplier ", mult, "x failed to integrate (retcode=", sol.retcode, "), skipping")
        continue
    end
    label = "vem_$(mult)x"
    for (name, extractor) in pairs(OBSERVED_SPECIES)
        raw = [extractor(sol.u[j]) for j in eachindex(sol.u)]
        df_vem[!, "$(name)_$(label)_norm"] = min_max_normalize(raw)
    end
end
CSV.write("results/vem_dose_response.csv", df_vem)
println("  Wrote results/vem_dose_response.csv")

println()
println("Done. tram_dose_response.csv (lower-risk sweep) and vem_dose_response.csv")
println("(higher-risk sweep, NN never saw varying Vemurafenib concentration during training).")
