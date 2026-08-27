#=
Simulates the best-fit HNODE model on a fine time grid (for smooth curves) and at the 6
real timepoints (for direct data comparison), and exports everything needed to reproduce
MATLAB-style fit plots -- without needing any plotting package on the cluster itself.

Produces two CSV files in results/:
  - model_fit_fine.csv    : smooth model trajectory, ~200 points from 0-48h, all 12 species
  - model_vs_data_6pt.csv : model output AND real experimental data at the 6 actual timepoints

Either plot these locally (Python/matplotlib, R, Excel, Julia+Plots on your own machine),
or share both CSVs back in the conversation for plots to be generated directly.
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

# TODO: must match Step 2a's winning architecture (trial 57: 2 layers, width 4)
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

fitted_pars = best.parameters_training  # ComponentArray with .p_net and .ode_par (absolute values)

# ---- Fine grid simulation (smooth curves) ----
tspan = (initial_time_training_mapk, end_time_training_mapk)
tfine_minutes = range(tspan[1], tspan[2], length=200)
prob_fine = ODEProblem{true}(hnode_derivative_function, original_u0_mapk, tspan)
sol_fine = solve(prob_fine, integrator, p=fitted_pars, saveat=tfine_minutes, abstol=abstol, reltol=reltol)

if sol_fine.retcode != SciMLBase.ReturnCode.Success
    error("Fine-grid simulation failed with retcode = $(sol_fine.retcode)")
end

fine_df = DataFrame(t_hours = collect(tfine_minutes) ./ 60.0)
for (name, extractor) in pairs(OBSERVED_SPECIES)
    raw = [extractor(sol_fine.u[j]) for j in eachindex(sol_fine.u)]
    fine_df[!, "$(name)_raw"] = raw
    fine_df[!, "$(name)_norm"] = min_max_normalize(raw)
end
CSV.write("results/model_fit_fine.csv", fine_df)
println("Wrote results/model_fit_fine.csv (", nrow(fine_df), " timepoints x ", ncol(fine_df), " columns)")

# ---- 6-timepoint simulation (direct data comparison) ----
sol_6pt = solve(prob_fine, integrator, p=fitted_pars, saveat=timestamps_minutes_mapk, abstol=abstol, reltol=reltol)

if sol_6pt.retcode != SciMLBase.ReturnCode.Success
    error("6-timepoint simulation failed with retcode = $(sol_6pt.retcode)")
end

compare_df = DataFrame(t_hours = timestamps_hours_mapk)
for (name, extractor) in pairs(OBSERVED_SPECIES)
    raw = [extractor(sol_6pt.u[j]) for j in eachindex(sol_6pt.u)]
    model_norm = min_max_normalize(raw)
    compare_df[!, "$(name)_model_norm"] = model_norm
    compare_df[!, "$(name)_experimental_norm"] = exp_data_norm_mapk[name]
end
CSV.write("results/model_vs_data_6pt.csv", compare_df)
println("Wrote results/model_vs_data_6pt.csv (", nrow(compare_df), " timepoints x ", ncol(compare_df), " columns)")

println()
println("Done. Download both CSVs from results/ and either plot locally, or share them")
println("back in the conversation for plots to be generated directly.")
