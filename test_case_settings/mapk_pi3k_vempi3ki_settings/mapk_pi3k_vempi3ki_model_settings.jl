#=
Settings for the Vemurafenib + PI3K-inhibitor training condition. This is a SEPARATE model
from the Vem+Tram fit (per the separate-models-per-condition design) -- it reuses the same
68-state mechanistic structure and the same HNODE paradox-activation term, but is fit
independently against this condition's own real experimental data.

Differences from the Vem+Tram condition, all handled here:
  - 9 observed species (not 12): pEGFR, pCRAF, pMEK, pERK, DUSP, pAKT, IGF1R, pHer2, pHer3.
    Missing vs Vem+Tram: panRAS, p4EBP1, pDGFR, pS6K. Added: IGF1R.
  - Trametinib is ABSENT (Tram.conc = 0); a PI3K inhibitor is present instead.
  - The PI3K-inhibition strength is a FREE parameter fit to this data (unlike the earlier
    alpelisib extrapolation, where it was fixed) -- this is the whole point of having real
    data for this condition.

This file is included AFTER mapk_pi3k_vemtram_model_settings.jl, and reuses its
original_parameters_mapk, original_u0_mapk, fixed_idx_mapk, etc. It only overrides/adds the
condition-specific pieces.
=#

# ---- Real experimental data for the Vem+PI3K condition ----
const exp_data_raw_vempi3ki = Dict(
    :pEGFR => [0.222379739, 0.622877159, 0.629217784, 0.533530834, 0.022513609, 0.010036399],
    :pCRAF => [0.234376572, 0.641878896, 0.567434544, 0.406320223, 0.582899195, 0.25113447],
    :pMEK  => [1.936660577, 0.029380652, 0.012873835, 0.03390921,  0.095155796, 0.944936578],
    :pERK  => [3.273353557, 0.075717978, 0.011570416, 0.00642985,  0.041863585, 0.91621491],
    :DUSP  => [2.854207662, 2.842703936, 1.163746208, 0.332720449, 0.030434242, 0.094073888],
    :pAKT  => [0.527301325, 0.14645732,  0.095895017, 0.0895019432, 0.0412820453, 0.0269891704],
    :IGF1R => [1.180034579, 0.967927178, 0.808905442, 0.781013289, 0.41928501,  0.870763253],
    :pHer2 => [0.306924546, 0.275751955, 0.32171108,  0.23070312,  1.013023288, 1.045536401],
    :pHer3 => [0.295284147, 0.285719072, 0.385045943, 0.582261781, 0.751301308, 0.264889608],
)

exp_data_norm_vempi3ki = Dict(k => min_max_normalize(v) for (k, v) in exp_data_raw_vempi3ki)

# ---- Observed species for THIS condition (9), with their state extractors ----
# IGF1R active receptor = u[40], following the pEGFR=u[3] / pHer2=u[6] / pHer3=u[9] pattern
# (the 3rd/active state in each receptor triplet). Confirmed by user.
const OBSERVED_SPECIES_VEMPI3KI = (
    pEGFR = u -> u[3],
    pCRAF = u -> u[23],
    pMEK  = u -> u[27],
    pERK  = u -> u[29],
    DUSP  = u -> u[31],
    pAKT  = u -> u[53],
    IGF1R = u -> u[40],
    pHer2 = u -> u[6],
    pHer3 = u -> u[9],
)

# ---- Per-species residual weights, in the SAME order as OBSERVED_SPECIES_VEMPI3KI ----
# Reusing Vem+Tram's weights for the shared species and assigning IGF1R a moderate weight.
# [pEGFR, pCRAF, pMEK, pERK, DUSP, pAKT, IGF1R, pHer2, pHer3]
const species_weights_vempi3ki = [3.0, 10.0, 10.0, 10.0, 2.0, 15.0, 5.0, 5.0, 5.0]

# ---- PI3K inhibitor parameterization ----
# Reuses the Hill-type inhibition on the PIP2->PIP3 catalytic step (mapk_pi3ki_extension_functions.jl).
# Because we're FITTING against real data, the inhibition STRENGTH is made a free parameter:
# we hold drug concentration and Hill coefficient fixed at reference values and let the
# optimizer learn an effective IC50 (equivalently, the effective inhibition fraction). See
# the HNODE function file for how pi3ki_ic50 becomes a fitted quantity.
const pi3ki_conc_ref = 5.62e-6      # reference concentration (alpelisib clinical Cmax scale); fixed
const pi3ki_hill_n_ref = 1.0        # fixed
const pi3ki_ic50_nominal = 4.6e-9   # nominal IC50 (alpelisib biochemical); FIT around this, +/- search range

# Vem+PI3K uses the same training time window as Vem+Tram
initial_time_training_vempi3ki = initial_time_training_mapk
end_time_training_vempi3ki = end_time_training_mapk
timestamps_minutes_vempi3ki = timestamps_minutes_mapk
