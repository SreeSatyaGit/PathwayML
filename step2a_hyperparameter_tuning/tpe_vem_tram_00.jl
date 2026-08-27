#=
Step 2a - Stage 1: TPE hyperparameter tuning for the Vem+Tram MAPK/PI3K HNODE model,
noise level 0.0. Adapted from HNODECB's tpe_cell_ap_00.jl / tpe_lv_00.jl, generalized to
loop over N mechanistic parameters instead of hardcoding p1..pN.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux

using PyCall
optuna = pyimport("optuna")

result_folder = "results_vem_tram"
if !isdir(result_folder)
    mkdir(result_folder)
end
result_name_string = "vem_tram_00.jld"
error_level = "e0.0"

include("../test_case_settings/vem_tram_model_settings/vem_tram_model_functions.jl")
include("../test_case_settings/vem_tram_model_settings/vem_tram_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5"]

# TODO: start with these stiff-solver settings (crosstalk models are often stiff); if the
# solver is taking a huge number of steps in early trials, this is already the right choice.
# If it's clearly non-stiff, switch to: integrator = Vern7(); sensealg = InterpolatingAdjoint(...)
integrator = TRBDF2(autodiff=false)
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

ode_data = deserialize("../datasets/e0.0/data/ode_data_vem_tram.jld")
ode_data_sd = deserialize("../datasets/e0.0/data/ode_data_std_vem_tram.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_vem_tram.jld")
solution_sd_dataframe = deserialize("../datasets/e0.0/data/pert_df_sd_vem_tram.jld")

rng = Random.default_rng()
Random.seed!(rng, 0)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT (Step 1) #####################################################

tspan = (initial_time_training, end_time_training)
tsteps = range(tspan[1], tspan[2], length=21)
datasize_full = length(tsteps)

shuffled_positions = shuffle(2:size(solution_dataframe)[1])
first_validation = rand(2:5)
validation_mask = [(first_validation + k * 5) for k in 0:3]
training_mask = [j for j in 1:size(solution_dataframe)[1] if !(j in validation_mask)]
training_mask = sort(training_mask)
validation_mask = sort(validation_mask)

original_ode_data = deepcopy(ode_data)
original_solution_dataframe = deepcopy(solution_dataframe)

ode_data = original_ode_data[:, training_mask]
solution_dataframe = original_solution_dataframe[training_mask, :]

validation_ode_data = original_ode_data[:, validation_mask]
validation_solution_dataframe = original_solution_dataframe[validation_mask, :]

######################################################################################################################################

# ---- Search space for the mechanistic parameters (TODO: tighten these once you have real
# priors -- 10^-2..10^2 x ground truth is intentionally wide, matching the paper's default) ----
n_mech_params = length(original_parameters)
lower_bounds = 10.0^-2 .* original_parameters
upper_bounds = 10.0^2 .* original_parameters

normalization_factor = maximum(original_ode_data, dims=2) - minimum(original_ode_data, dims=2)
normalization_factor_training = repeat(normalization_factor, 1, size(ode_data)[2])
normalization_factor_validation = repeat(normalization_factor, 1, size(validation_ode_data)[2])

tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
seed = abs(rand(rng, Int))

# NN input/output sizing -- must match view(u, [1,3,5]) and single output in the model function
nn_input_size = 3
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

function objective(trial)

    original_param_vector = [trial.suggest_float("p$i", lower_bounds[i], upper_bounds[i]) for i in 1:n_mech_params]

    learning_rate_adam = trial.suggest_float("learning_rate_adam", 1e-5, 1e-1, step=nothing, log=true)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4)
    num_hidden_nodes = trial.suggest_int("num_hidden_nodes", 2, 4)
    ms_group_size = trial.suggest_int("ms_group_size", 2, 10)
    ms_continuity_term = trial.suggest_float("ms_continuity_term", 0.001, 1000.0, step=nothing, log=true)

    approximating_neural_network = build_nn(num_hidden_layers, num_hidden_nodes)

    initial_time = time()
    rng_tmp = StableRNG(seed)
    local_approximating_neural_network = deepcopy(approximating_neural_network)
    p_net, st = Lux.setup(rng_tmp, local_approximating_neural_network)

    uode_derivative_function = get_uode_model_function(local_approximating_neural_network, st, original_param_vector)
    initial_parameters = ones(n_mech_params)

    prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)
    ranges = DiffEqFlux.group_ranges(datasize, ms_group_size)

    function loss_function(data, deviation, pred)
        original_cost = sum(abs2.(data .- pred) ./ abs2.(deviation))
        return 1 / size(data, 2) * sum(original_cost)
    end

    function loss_multiple_shooting(θ)
        p = θ.p
        tsteps_local = tmp_steps
        initial_point_parameters = θ.u0

        function unstable_check(dt, u, p, t)
            return any(abs.(u) .> 1e7)
        end

        sols = [
            solve(
                remake(prob_uode_pred; p=p, tspan=(tsteps_local[first(rg)], tsteps_local[last(rg)]), u0=initial_point_parameters[:, first(rg)]),
                integrator; saveat=tsteps_local[rg], reltol=reltol, abstol=abstol,
                sensealg=sensealg, unstable_check=unstable_check, verbose=false
            ) for rg in ranges
        ]

        for i in 1:length(sols)
            if size(Array(sols[i]))[2] != length(ranges[i])
                return Inf
            end
        end
        group_predictions = Array.(sols)

        loss = 0
        for (i, rg) in enumerate(ranges)
            u = ode_data[:, rg]
            std = normalization_factor_training[:, rg]
            û = group_predictions[i]
            loss += loss_function(u, std, û)
        end

        for (i, rg) in enumerate(ranges)
            i == 1 && continue
            u0 = group_predictions[i-1][:, end]
            u1 = group_predictions[i][:, 1]
            loss += ms_continuity_term * sum(abs2, u0 - u1)
        end

        return loss
    end

    epoch = 1
    function callback(θ, l, training_epochs, training_costs, num_epoch_to_finish, stuck)
        println("Epoch: " * string(epoch) * " - Loss: " * string(l))
        if epoch == num_epoch_to_finish
            return true
        end
        training_epochs[epoch] = epoch
        training_costs[epoch] = l
        if time() - initial_time > 2 * 60
            println("Too slow optimization")
            return true
        end
        if epoch > 10 && minimum(training_costs[(epoch-5):(epoch)]) > 10^6
            stuck[1] = true
            return true
        end
        epoch += 1
        return false
    end

    function validation_loss_function(θ)
        loss = 0.0
        try
            validation_df = validation_solution_dataframe
            max_time = extrema(validation_df.t)[2]
            par = θ.p
            init_par = θ.u0
            prob = remake(prob_uode_pred; p=par, tspan=(0, max_time), u0=init_par[:, 1])

            function unstable_check(dt, u, p, t)
                return any(abs.(u) .> 1e7)
            end

            solutions = solve(prob, integrator, p=par, saveat=validation_df.t, abstol=abstol, reltol=reltol,
                sensealg=sensealg, unstable_check=unstable_check, verbose=false)
            x = Array(solutions)
            if size(x)[2] != size(validation_df)[1]
                return Inf
            end
            loss = 1 / size(validation_df, 1) * sum(abs2.(Array(validation_df[:, 2:end])' .- x) ./ abs2.(normalization_factor_validation))
        catch
            loss = Inf
        end
        return loss
    end

    adtype = Optimization.AutoZygote()

    p_net = ComponentArray(p_net)
    ode_par = ComponentArray(initial_parameters)
    p = ComponentArray{eltype(p_net)}()
    p = ComponentArray(p; p_net)
    p = ComponentArray(p; ode_par)
    u0 = deepcopy(ode_data)
    starting_point_in = ComponentVector{Float64}(p=p, u0=u0)

    training_epochs = zeros(Int, 50000)
    training_costs = zeros(50000)

    optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, starting_point_in)
    opt = OptimizationOptimisers.Adam(learning_rate_adam)

    stuck = [false]
    res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, 20, stuck), maxiters=501)

    validation_resulting_cost = stuck[1] ? Inf : validation_loss_function(res.u)

    θ_to_memorize = deepcopy(res.u)
    global trial_parameters
    push!(trial_parameters, θ_to_memorize)
    ode_par_optimized = res.u.p.ode_par .* original_param_vector

    study.tell(trial, validation_resulting_cost)

    result_trial = nothing
    try
        result_params = copy(trial.params)
        for i in 1:n_mech_params
            result_params["p$i"] = min(max(ode_par_optimized[i], lower_bounds[i]), upper_bounds[i])
        end
        result_trial = optuna.create_trial(params=result_params, distributions=trial.distributions, value=validation_resulting_cost)
        study.add_trial(result_trial)
    catch
        println("failed attempt")
    end

    return result_trial
end

global trial_parameters = []
study = optuna.create_study(sampler=optuna.samplers.TPESampler(consider_prior=false, n_startup_trials=200, multivariate=true, seed=0))
for optuna_iteration in 1:10
    trial = study.ask()
    res_trial = objective(trial)
end

result = (study=study, trial_parameters=trial_parameters, error_level=error_level)
serialize(result_folder * "/" * result_name_string, result)
