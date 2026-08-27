#=
Run this AFTER tpe_mapk_hnode_00.jl finishes. Loads the serialized Optuna study,
finds the best trial, and prints exactly what to paste into
step2b_model_trainer/train_mapk_hnode_00.jl's "TUNED HYPERPARAMETERS" block.
=#

cd(@__DIR__)
using Serialization, PyCall, PyCall, ComponentArrays

result = deserialize("results_mapk_hnode/mapk_hnode_00.jld")
study = result.study

all_trials = study.trials
println("Total trials recorded: ", length(all_trials))
println()

# Don't trust study.best_trial here -- pull every trial's (number, value, state) directly
# and find the true minimum finite value ourselves.
trial_info = [(number=t.number, value=t.value, state=string(t.state)) for t in all_trials]

finite_trials = filter(t -> t.value !== nothing && isfinite(t.value), trial_info)
println("Trials with a finite recorded value: ", length(finite_trials), " / ", length(trial_info))

if isempty(finite_trials)
    error("No trials have a finite value -- something is still wrong upstream of this script.")
end

sorted_trials = sort(finite_trials, by = t -> t.value)
println()
println("Top 5 trials by lowest loss:")
for t in sorted_trials[1:min(5, end)]
    println("  trial ", t.number, "  loss = ", t.value, "  state = ", t.state)
end
println()

best = sorted_trials[1]
best_trial_number = best.number
println("Using trial ", best_trial_number, " (loss = ", best.value, ") as the best trial.")
println()

best_trial = first(t for t in all_trials if t.number == best_trial_number)
params = best_trial.params

println("=== Architecture ===")
println("num_hidden_layers = ", params["num_hidden_layers"])
println("num_hidden_nodes  = ", params["num_hidden_nodes"], "  (width = 2^", params["num_hidden_nodes"], " = ", 2^params["num_hidden_nodes"], ")")
println()

println("=== Optimizer settings ===")
println("learning_rate_adam = ", params["learning_rate_adam"])
println("l2_reg_weight       = ", params["l2_reg_weight"])
println()

println("=== Mechanistic parameter starting values (paste as pretrained_ode_pars) ===")
n_mech_params = 60
p_values = [params["p$i"] for i in 1:n_mech_params]
println("pretrained_ode_pars = [")
for (i, v) in enumerate(p_values)
    print("    ", v)
    print(i < n_mech_params ? ",\n" : "\n")
end
println("]")
println()

println("=== Reminder: build the NN architecture in train_mapk_hnode_00.jl to match ===")
width = 2^params["num_hidden_nodes"]
nlayers = params["num_hidden_layers"]
println("Lux.Chain(")
println("    Lux.Dense(2, $width, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),")
for _ in 1:(nlayers - 1)
    println("    Lux.Dense($width, $width, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),")
end
println("    Lux.Dense($width, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),")
println(")")