#=
Step 2a: TPE hyperparameter tuning for the Vem+PI3K training condition.

Same methodology and deviations as the Vem+Tram Step 2a (single-shooting, no held-out
validation split -- 6 timepoints). Condition-specific differences:
  - 9 observed species (see OBSERVED_SPECIES_VEMPI3KI), not 12
  - Trametinib absent (Tram.conc set to 0 in fixed values)
  - PI3K inhibitor active, with a FITTABLE effective IC50 as an EXTRA free parameter
    (so the mechanistic parameter vector has length 60 + 1 = 61 here)
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, SciMLBase, LinearAlgebra, Random, DataFrames, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux

using PyCall
optuna = pyimport("optuna")

result_folder = "results_vempi3ki_hnode"
if !isdir(result_folder)
    mkdir(result_folder)
end
result_name_string = "vempi3ki_hnode_00.jld"

include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_functions.jl")
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_hnode_functions.jl")   # for assemble_full_parameter_vector
include("../test_case_settings/mapk_pi3k_vemtram_settings/mapk_pi3k_vemtram_model_settings.jl")     # base params, ICs, fixed idx, bounds
include("../test_case_settings/mapk_pi3k_vempi3ki_settings/mapk_pi3k_vempi3ki_hnode_functions.jl")  # get_uode_model_function_vempi3ki
include("../test_case_settings/mapk_pi3k_vempi3ki_settings/mapk_pi3k_vempi3ki_model_settings.jl")   # this condition's data, species, weights

integrator = TRBDF2(autodiff=false)
abstol = 1e-8
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

rng = Random.default_rng()
Random.seed!(rng, 0)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

n_mech_params = length(free_idx_hnode_mapk)   # 60 shared mechanistic
n_total_params = n_mech_params + 1            # + 1 fittable PI3K IC50
tspan = (initial_time_training_vempi3ki, end_time_training_vempi3ki)

# Trametinib off for this condition
fixed_values_vempi3ki = copy(fixed_values_mapk)
fixed_values_vempi3ki[1] = 0.0   # Tram.conc = 0

# Species order / extractors / weights come from the condition settings file
const species_order = (:pEGFR, :pCRAF, :pMEK, :pERK, :DUSP, :pAKT, :IGF1R, :pHer2, :pHer3)
species_extractors = OBSERVED_SPECIES_VEMPI3KI

min_max_normalize_for_loss(v) = (v .- minimum(v)) ./ (maximum(v) - minimum(v) + 1e-6)

nn_input_size = 2
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

function weighted_loss(sol, retcode_ok::Bool)
    if !retcode_ok || length(sol.t) != length(timestamps_minutes_vempi3ki)
        return Inf
    end
    total = 0.0
    for (i, name) in enumerate(species_order)
        extractor = species_extractors[name]
        raw = [extractor(sol.u[j]) for j in eachindex(sol.u)]
        model_norm = min_max_normalize_for_loss(raw)
        exp_norm = exp_data_norm_vempi3ki[name]
        total += species_weights_vempi3ki[i] * sum(abs2, model_norm .- exp_norm)
    end
    return total
end

function objective(trial)
    # 60 shared mechanistic parameters (within +/-1.5x nominal, as in Vem+Tram)
    original_param_vector = [trial.suggest_float("p$i", lower_bounds_hnode_mapk[i], upper_bounds_hnode_mapk[i]) for i in 1:n_mech_params]
    # + 1 fittable PI3K IC50 scale factor (search 0.01x .. 100x of nominal IC50)
    pi3ki_ic50_scale = trial.suggest_float("pi3ki_ic50_scale", 1e-2, 1e2, step=nothing, log=true)

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

    hnode_derivative_function = get_uode_model_function_vempi3ki(
        local_approximating_neural_network, st, original_param_vector,
        free_idx_hnode_mapk, fixed_idx_mapk, fixed_values_vempi3ki,
        pi3ki_conc_ref, pi3ki_hill_n_ref, pi3ki_ic50_nominal
    )
    # ode_par = [60 mechanistic scale factors (init 1.0)] + [PI3K IC50 scale factor (init to sampled)]
    initial_ode_par = vcat(ones(n_mech_params), pi3ki_ic50_scale)

    prob = ODEProblem{true}(hnode_derivative_function, original_u0_mapk, tspan)

    function unstable_check(dt, u, p, t)
        return any(abs.(u) .> 1e7)
    end

    function loss_fn(θ)
        par = θ.p
        local sol
        try
            sol = solve(prob, integrator; p=par, saveat=timestamps_minutes_vempi3ki,
                        abstol=abstol, reltol=reltol, sensealg=sensealg,
                        unstable_check=unstable_check, verbose=false)
        catch e
            return 1e6
        end
        data_loss = weighted_loss(sol, sol.retcode == SciMLBase.ReturnCode.Success)
        if !isfinite(data_loss)
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
        flush(stdout)
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

    final_theta = best_theta[] === nothing ? res.u : best_theta[]
    final_cost = best_cost[]

    global trial_parameters
    push!(trial_parameters, deepcopy(final_theta))
    study.tell(trial, final_cost)

    return nothing
end

global trial_parameters = []
study = optuna.create_study(sampler=optuna.samplers.TPESampler(consider_prior=false, n_startup_trials=50, multivariate=true, seed=0))

n_trials = 100
for optuna_iteration in 1:n_trials
    trial = study.ask()
    try
        objective(trial)
    catch ex
        println("TRIAL ", optuna_iteration, " CRASHED (caught, continuing to next trial):")
        showerror(stdout, ex)
        println()
        flush(stdout)
        try
            study.tell(trial, 1e6)
        catch
        end
    end
end

result = (study=study, trial_parameters=trial_parameters)
serialize(result_folder * "/" * result_name_string, result)
