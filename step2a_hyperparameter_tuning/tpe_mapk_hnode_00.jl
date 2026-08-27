#=
Step 2a: TPE hyperparameter tuning for the real MAPK/PI3K + Vem/Tram HNODE model
(paradoxical RAF activation replaced by a neural network).

DEVIATIONS FROM THE PAPER'S METHODOLOGY (both due to having only 6 real timepoints):
  1. SINGLE-SHOOTING, not multiple shooting. With only 6 unevenly-spaced timepoints
     (0, 1, 4, 8, 24, 48h), there isn't enough data to segment meaningfully -- the
     whole trajectory is solved in one shot per loss evaluation.
  2. NO held-out validation split. Splitting 6 points further leaves too little
     signal in either set. The TPE objective here is the FULL-DATA training loss
     itself, not a validation loss. This means hyperparameter selection is less
     rigorous than the paper's approach -- worth revisiting if you get more
     experimental timepoints later (e.g. from a follow-up dose/time study).

The loss function replicates your MATLAB objectiveFunction_all exactly: simulate
the full 48h trajectory, min-max normalize each of the 12 observed species
(using the SAME epsilon=1e-6 convention your MATLAB fitting objective used, not
the eps() convention used for display/data-prep), and compute the per-species
weighted sum of squared residuals against the real experimental data.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, Random, DataFrames, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux

using PyCall
optuna = pyimport("optuna")

result_folder = "results_mapk_hnode"
if !isdir(result_folder)
    mkdir(result_folder)
end
result_name_string = "mapk_hnode_00.jld"

include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_settings.jl")

integrator = TRBDF2(autodiff=false)
abstol = 1e-8
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

rng = Random.default_rng()
Random.seed!(rng, 0)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

n_mech_params = length(free_idx_hnode_mapk)   # 60
tspan = (initial_time_training_mapk, end_time_training_mapk)

# species order and weights, matching MATLAB exactly
const species_order = (:pEGFR, :panRAS, :pMEK, :pERK, :DUSP, :pCRAF, :pAKT, :p4EBP1, :pHer2, :pHer3, :pDGFR, :pS6K)
const species_extractors = (
    pEGFR = OBSERVED_SPECIES.pEGFR, panRAS = OBSERVED_SPECIES.panRAS, pMEK = OBSERVED_SPECIES.pMEK,
    pERK = OBSERVED_SPECIES.pERK, DUSP = OBSERVED_SPECIES.DUSP, pCRAF = OBSERVED_SPECIES.pCRAF,
    pAKT = OBSERVED_SPECIES.pAKT, p4EBP1 = OBSERVED_SPECIES.p4EBP1, pHer2 = OBSERVED_SPECIES.pHer2,
    pHer3 = OBSERVED_SPECIES.pHer3, pDGFR = OBSERVED_SPECIES.pDGFR, pS6K = OBSERVED_SPECIES.pS6K,
)
# fitting-objective normalization uses eps=1e-6, matching MATLAB's objectiveFunction_all
# (distinct from the eps() used for display/data-prep elsewhere)
min_max_normalize_for_loss(v) = (v .- minimum(v)) ./ (maximum(v) - minimum(v) + 1e-6)

nn_input_size = 2   # [dimer level, Vemurafenib concentration]
nn_output_size = 1

function build_nn(num_hidden_layers, num_hidden_nodes)
    width = 2^num_hidden_nodes
    layers = Any[Lux.Dense(nn_input_size, width, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform)]
    for _ in 1:(num_hidden_layers - 1)
        push!(layers, Lux.Dense(width, width, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform))
    end
    push!(layers, Lux.Dense(width, nn_output_size; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform))
    return Lux.Chain(layers...)
end

"""
Weighted sum-of-squares loss, replicating MATLAB's objectiveFunction_all: solve
once over the full window, normalize each species trajectory, weight, and sum.
Returns Inf if the integration fails (matching MATLAB's catch-block fallback).
"""
function weighted_loss(sol, retcode_ok::Bool)
    if !retcode_ok || length(sol.t) != length(timestamps_minutes_mapk)
        return Inf
    end
    total = 0.0
    for (i, name) in enumerate(species_order)
        extractor = species_extractors[name]
        raw = [extractor(sol.u[j]) for j in eachindex(sol.u)]
        model_norm = min_max_normalize_for_loss(raw)
        exp_norm = exp_data_norm_mapk[name]
        total += species_weights_mapk[i] * sum(abs2, model_norm .- exp_norm)
    end
    return total
end

function objective(trial)
    original_param_vector = [trial.suggest_float("p$i", lower_bounds_hnode_mapk[i], upper_bounds_hnode_mapk[i]) for i in 1:n_mech_params]

    learning_rate_adam = trial.suggest_float("learning_rate_adam", 1e-5, 1e-1, step=nothing, log=true)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
    num_hidden_nodes = trial.suggest_int("num_hidden_nodes", 2, 4)
    l2_reg_weight = trial.suggest_float("l2_reg_weight", 1e-6, 1e-1, step=nothing, log=true)

    approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)

    initial_time = time()
    seed = abs(rand(rng, Int))
    rng_tmp = StableRNG(seed)
    local_approximating_neural_network = deepcopy(approximating_neural_network)
    p_net, st = Lux.setup(rng_tmp, local_approximating_neural_network)

    hnode_derivative_function = get_uode_model_function(
        local_approximating_neural_network, st, original_param_vector,
        free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_mapk
    )
    initial_ode_par = ones(n_mech_params)

    prob = ODEProblem{true}(hnode_derivative_function, original_u0_mapk, tspan)

    function unstable_check(dt, u, p, t)
        return any(abs.(u) .> 1e7)
    end

    function loss_fn(θ)
        par = θ.p
        local sol
        try
            sol = solve(prob, integrator; p=par, saveat=timestamps_minutes_mapk,
                        abstol=abstol, reltol=reltol, sensealg=sensealg,
                        unstable_check=unstable_check, verbose=false)
        catch e
            println("INTEGRATION EXCEPTION: ", sprint(showerror, e))
            flush(stdout)
            return 1e6
        end
        data_loss = weighted_loss(sol, sol.retcode == SciMLBase.ReturnCode.Success)
        if !isfinite(data_loss)
            println("NON-FINITE LOSS: retcode=", sol.retcode, " npoints=", length(sol.t), " expected=", length(timestamps_minutes_mapk))
            flush(stdout)
            return 1e6
        end
        reg_loss = l2_reg_weight * sum(abs2, θ.p.p_net)
        return data_loss + reg_loss
    end

    epoch = 1
    best_theta = Ref{Any}(nothing)
    best_cost = Ref(Inf)
    function callback(θ, l, training_epochs, training_costs, num_epoch_to_finish, stuck)
        println("Epoch: " * string(epoch) * " - Loss: " * string(l))
        if isfinite(l) && l < best_cost[]
            best_cost[] = l
            best_theta[] = deepcopy(θ)
        end
        if epoch == num_epoch_to_finish
            return true
        end
        training_epochs[epoch] = epoch
        training_costs[epoch] = l
        if time() - initial_time > 2 * 60
            println("Too slow optimization")
            return true
        end
        if epoch > 10 && minimum(training_costs[(epoch-5):(epoch)]) > 1e5
            stuck[1] = true
            return true
        end
        epoch += 1
        return false
    end

    adtype = Optimization.AutoZygote()

    p_net_ca = ComponentArray(p_net)
    ode_par_ca = ComponentArray(initial_ode_par)
    p = ComponentArray{eltype(p_net_ca)}()
    p = ComponentArray(p; p_net=p_net_ca)
    p = ComponentArray(p; ode_par=ode_par_ca)
    starting_point_in = ComponentVector{Float64}(p=p)

    training_epochs = zeros(Int, 5000)
    training_costs = zeros(5000)

    optf = Optimization.OptimizationFunction((x, p) -> loss_fn(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, starting_point_in)
    opt = OptimizationOptimisers.Adam(learning_rate_adam)

    stuck = [false]
    res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, 300, stuck), maxiters=300)

    # Use the BEST checkpoint reached during training, not wherever Adam happened to end
    # up (which may have since drifted into an unstable region -- very possible over 300
    # unconstrained gradient steps on this sensitive, stiff system). Falls back to the
    # final state only in the edge case where literally no epoch produced a finite loss.
    final_theta = best_theta[] === nothing ? res.u : best_theta[]
    final_cost = best_cost[]

    global trial_parameters
    push!(trial_parameters, deepcopy(final_theta))

    study.tell(trial, final_cost)

    return nothing
end

global trial_parameters = []
study = optuna.create_study(sampler=optuna.samplers.TPESampler(consider_prior=false, n_startup_trials=50, multivariate=true, seed=0))

# NOTE: full run should be more trials (e.g. 200-500); start smaller given only 6
# data points constrain how much a search can meaningfully explore.
n_trials = 100
for optuna_iteration in 1:n_trials
    trial = study.ask()
    res_trial = objective(trial)
end

result = (study=study, trial_parameters=trial_parameters)
serialize(result_folder * "/" * result_name_string, result)