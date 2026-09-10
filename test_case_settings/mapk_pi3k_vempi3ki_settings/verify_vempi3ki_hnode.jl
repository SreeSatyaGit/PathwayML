#=
Sanity check for the Vem+PI3K HNODE setup before running Step 2a. Builds a small untrained
NN, wires in the fittable-IC50 PI3K term with Trametinib off, confirms the 68-state system
integrates, and prints the (untrained) model vs. this condition's real data for all 9
observed species. Untrained -> not expected to match; only confirms the plumbing works.
=#

cd(@__DIR__)
using DifferentialEquations, SciMLBase, Lux, ComponentArrays, Random, StableRNGs, Printf

include("../mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")
include("../mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_settings.jl")
include("mapk_pi3k_vempi3ki_hnode_functions.jl")
include("mapk_pi3k_vempi3ki_model_settings.jl")

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
    Lux.Dense(2, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

rng = StableRNG(0)
p_net, st = Lux.setup(rng, approximating_neural_network)

n_mech_params = length(free_idx_hnode_mapk)
original_parameters_opt = original_parameters_mapk[free_idx_hnode_mapk]

fixed_values_vempi3ki = copy(fixed_values_mapk)
fixed_values_vempi3ki[1] = 0.0   # Trametinib off

hnode_fn = get_uode_model_function_vempi3ki(
    approximating_neural_network, st, original_parameters_opt,
    free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_vempi3ki,
    pi3ki_conc_ref, pi3ki_hill_n_ref, pi3ki_ic50_nominal
)

# ode_par: 60 mechanistic (init 1.0) + 1 PI3K IC50 scale (init 1.0 = nominal)
initial_ode_par = vcat(ones(n_mech_params), 1.0)
p_net_ca = ComponentArray(p_net)
ode_par_ca = ComponentArray(initial_ode_par)
p = ComponentArray{eltype(p_net_ca)}()
p = ComponentArray(p; p_net=p_net_ca)
p = ComponentArray(p; ode_par=ode_par_ca)

tspan = (initial_time_training_vempi3ki, end_time_training_vempi3ki)
prob = ODEProblem{true}(hnode_fn, original_u0_mapk, tspan, p)

println("Integrating Vem+PI3K HNODE (untrained NN, nominal params)...")
sol = solve(prob, TRBDF2(autodiff=false), saveat=timestamps_minutes_vempi3ki, abstol=1e-8, reltol=1e-6)

if sol.retcode != SciMLBase.ReturnCode.Success
    error("Integration failed with retcode = $(sol.retcode).")
end

println("Integration succeeded. Retcode: ", sol.retcode)
println()
for (name, extractor) in pairs(OBSERVED_SPECIES_VEMPI3KI)
    raw = [extractor(sol.u[j]) for j in eachindex(sol.u)]
    normalized = min_max_normalize(raw)
    println("--- $name ---")
    println("  model (untrained, norm): ", round.(normalized, digits=3))
    println("  experimental (norm):     ", round.(exp_data_norm_vempi3ki[name], digits=3))
end
println()
println("Free mechanistic params: ", n_mech_params, " + 1 fittable PI3K IC50 = ", n_mech_params + 1)
println("Plumbing OK if integration succeeded. Next: Step 2a (tpe_vempi3ki_hnode_00.jl).")
