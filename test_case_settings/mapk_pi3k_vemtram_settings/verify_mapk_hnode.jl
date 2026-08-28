#=
Sanity check for the HNODE variant (paradoxical RAF activation replaced by a
neural network). Builds a small, randomly-initialized NN, assembles the HNODE
derivative function with the 60 free mechanistic parameters at their nominal
(scale=1.0) values, and confirms the whole system integrates successfully.

A randomly-initialized NN will NOT reproduce the original paradox_activation
term correctly -- this script only confirms the plumbing (NN input/output
wiring, full-parameter-vector reconstruction, ODE integration) works, not that
the dynamics are correct yet. That comes after Step 2a/2b training.
=#

cd(@__DIR__)
using DifferentialEquations, SciMLBase, Lux, ComponentArrays, Random, StableRNGs, Printf

include("mapk_pi3k_vemtram_model_functions.jl")   # for OBSERVED_SPECIES, min_max_normalize
include("mapk_pi3k_vemtram_hnode_functions.jl")
include("mapk_pi3k_vemtram_model_settings.jl")

println("Building a small NN for the paradox-activation term...")
println("  inputs: [dimer level (u[62]), Vemurafenib concentration]  -> output: 1 (activation rate)")

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
    Lux.Dense(2, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

rng = StableRNG(0)
p_net, st = Lux.setup(rng, approximating_neural_network)

# mechanistic parameters at their nominal values (scale factor = 1.0 relative to
# original_parameters_mapk, matching how Step 2a/2b will parameterize the optimizer)
original_parameters_opt = original_parameters_mapk[free_idx_hnode_mapk]
initial_ode_par = ones(length(free_idx_hnode_mapk))

hnode_derivative_function = get_uode_model_function(
    approximating_neural_network, st, original_parameters_opt,
    free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_mapk
)

p_net_ca = ComponentArray(p_net)
ode_par_ca = ComponentArray(initial_ode_par)
p = ComponentArray{eltype(p_net_ca)}()
p = ComponentArray(p; p_net=p_net_ca)
p = ComponentArray(p; ode_par=ode_par_ca)

tspan = (initial_time_training_mapk, end_time_training_mapk)
prob = ODEProblem{true}(hnode_derivative_function, original_u0_mapk, tspan, p)

println("Integrating HNODE system (untrained NN -- dynamics will look wrong, that's expected)...")
sol = solve(prob, TRBDF2(autodiff=false), saveat=timestamps_minutes_mapk, abstol=1e-8, reltol=1e-6)

if sol.retcode != SciMLBase.ReturnCode.Success
    error("HNODE integration failed with retcode = $(sol.retcode). Check the NN " *
          "input/output wiring and full-parameter-vector reconstruction in " *
          "mapk_pi3k_vemtram_hnode_functions.jl before proceeding to Step 2a.")
end

println("HNODE integration succeeded. Retcode: ", sol.retcode)
println()
println("Free mechanistic parameters: ", length(free_idx_hnode_mapk), " (was 61, kParadoxCRAF now handled by NN)")
println("NN parameter count: ", length(p_net_ca))
println()
println("This confirms the plumbing works. Next: Step 2a hyperparameter tuning to")
println("actually fit both the 60 mechanistic parameters and the NN weights against")
println("the real experimental data.")
