#=
Derivative functions for the MAPK/PI3K crosstalk model under Vemurafenib + Trametinib treatment.

*** ILLUSTRATIVE TEMPLATE ***
The kinetic forms below (mass-action / simple saturable terms) and the choice of which
term is "unknown" are placeholders that mirror the structure used in the HNODECB paper's
test cases. Replace the equations, not just the parameter values, with whatever kinetic
laws you've derived/validated for your own model before trusting any results.

State ordering: u = [pRAF, pMEK, pERK, pPI3K, pAKT]
=#

"""
    ground_truth_function(du, u, p, t)

Full mechanistic model, INCLUDING the AKT -> RAF bypass-reactivation crosstalk term.
Used only to generate synthetic in-silico data to validate the pipeline end-to-end
before applying it to real data (same self-test strategy as the HNODECB paper).
"""
function ground_truth_function(du, u, p, t)
    k_raf, d_raf, k_mek, d_mek, k_erk, d_erk, k_pi3k, d_pi3k, k_akt, d_akt, k_cross = p

    I_vem  = vem_concentration(t)  / (vem_concentration(t)  + IC50_vem)
    I_tram = tram_concentration(t) / (tram_concentration(t) + IC50_tram)

    du[1] = k_raf * (1 - I_vem) * (1 - u[1]) - d_raf * u[1] + k_cross * u[5] * (1 - u[1])
    du[2] = k_mek * (1 - I_tram) * u[1] * (1 - u[2]) - d_mek * u[2]
    du[3] = k_erk * u[2] * (1 - u[3]) - d_erk * u[3]
    du[4] = k_pi3k * (1 - u[4]) - d_pi3k * u[4] * (1 + u[3])
    du[5] = k_akt * u[4] * (1 - u[5]) - d_akt * u[5]
end

"""
    get_uode_model_function(appr_neural_network, state, original_parameters_opt)

HNODE version: the AKT -> RAF crosstalk term (`k_cross * u[5] * (1 - u[1])` above) is
replaced by a neural network taking [pRAF, pERK, pAKT] as inputs -- restricted to the
species we assume are mechanistically plausible drivers of the unknown reactivation term,
per the paper's approach of not feeding the NN the full state unnecessarily.

Mechanistic parameters are scaled by `original_parameters_opt`, the per-parameter starting
point sampled during Step 2a hyperparameter tuning (so the optimizer works in normalized
[~1] units regardless of each parameter's absolute scale).
"""
function get_uode_model_function(appr_neural_network, state, original_parameters_opt)
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters_opt = original_parameters_opt

            ode_par = p.ode_par .* original_parameters_opt
            k_raf, d_raf, k_mek, d_mek, k_erk, d_erk, k_pi3k, d_pi3k, k_akt, d_akt = ode_par

            I_vem  = vem_concentration(t)  / (vem_concentration(t)  + IC50_vem)
            I_tram = tram_concentration(t) / (tram_concentration(t) + IC50_tram)

            # NN sees only [pRAF, pERK, pAKT] -- restrict inputs to biologically plausible drivers
            û = appr_neural_network(view(u, [1, 3, 5]), p.p_net, st)[1]

            @inbounds du[1] = k_raf * (1 - I_vem) * (1 - u[1]) - d_raf * u[1] + û[1]
            @inbounds du[2] = k_mek * (1 - I_tram) * u[1] * (1 - u[2]) - d_mek * u[2]
            @inbounds du[3] = k_erk * u[2] * (1 - u[3]) - d_erk * u[3]
            @inbounds du[4] = k_pi3k * (1 - u[4]) - d_pi3k * u[4] * (1 + u[3])
            @inbounds du[5] = k_akt * u[4] * (1 - u[5]) - d_akt * u[5]
        end
end
