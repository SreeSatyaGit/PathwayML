#=
Settings for the real 68-state MAPK/PI3K + Vemurafenib/Trametinib model, ported
directly from the user's MATLAB script. All values below are the user's real,
already-characterized values -- nothing here is a placeholder.
=#

# ---- Full 68-element parameter vector (same order as MATLAB params_vector) ----
const original_parameters_mapk = Float64[
    10e-5, 10e-6, 10e-3,                      # 1-3:   ka1, kr1, kc1
    10e-4, 7e-6, 20e-5,                       # 4-6:   kpCraf, kpMek, kpErk
    10e-7, 20e-6, 10e-5, 7e-7, 36e-6,         # 7-11:  kDegradEgfr, kErkInbEgfr, kShcDephos, kptpDeg, kGrb2CombShc
    10e-5, 10e-5, 17e-6,                      # 12-14: kSprtyInbGrb2, kSosCombGrb2, kErkPhosSos
    15.7342e-6, 1.7342e-4, 20.7342e-5, 1.7342e-8,  # 15-18: kErkPhosPcraf, kPcrafDegrad, kErkPhosMek, kMekDegrad
    2.7342e-4, 1.7342e-7, 10e-2, 1.7342e-04, 1.11e-8,  # 19-23: kDuspInbErk, kErkDeg, kinbBraf, kDuspStop, kDusps
    1.7e-06, 5.5000e-5, 10e-7, 8e-5,           # 24-27: kSproutyForm, kSprtyComeDown, kdegrad, km_Sprty_decay
    1.11e-8, 1.7e-06,                          # 28-29: km_Dusp, km_Sprty
    1e-5, 1e-6,                                # 30-31: kErkDephos, kDuspDeg
    10e-5, 10e-5,                              # 32-33: kHer2_act, kHer3_act
    10e-4, 10e-4, 10e-4, 10e-4,                # 34-37: k_p85_bind_EGFR/Her2/Her3/IGFR
    10e-5, 10e-4,                              # 38-39: k_p85_unbind, k_PI3K_recruit
    10e-4,                                     # 40:    kMTOR_Feedback
    10e-4, 10e-5,                              # 41-42: k_PIP2_to_PIP3, k_PTEN
    10e-5, 10e-7,                              # 43-44: kAkt, kdegradAKT
    10e-8, 10e-3, 10e-5, 10e-5,                # 45-48: kb1, k43b1, k4ebp1, k_4EBP1_dephos
    5e-6, 5e-6,                                # 49-50: kKSRphos, kKSRdephos
    8e-6, 8e-6, 3e-6,                          # 51-53: kMekByBraf, kMekByCraf, kMekByKSR
    1e-6, 1e-9, 1e-9, 2.0,                     # 54-57: Tram.conc, Ki_RAF, Ki_KSR, Hill_n (FIXED)
    1.0, 6e-6, 1e-5,                           # 58-60: Vem.conc (FIXED), kDimerForm, kDimerDissoc
    0.5,                                       # 61:    Paradox.gamma
    0.4, 1.5,                                  # 62-63: Vem.IC50 (FIXED), Vem.Hill_n (FIXED)
    10e-5, 10e-4,                              # 64-65: kPDGFR_act, k_p85_bind_PDGFR
    10e-5, 10e-5,                              # 66-67: kS6K_phos, kS6K_dephos
    0.05,                                      # 68:    K_displace
]
@assert length(original_parameters_mapk) == 68 "Expected 68 parameters, got $(length(original_parameters_mapk))"

# ---- Fixed vs free parameters (1-indexed, matches MATLAB fixedIdx exactly) ----
# Fixed: drug dose, IC50, Ki (binding affinity), Hill coefficients -- known/measured,
# not fitted. K_displace (68) stays FREE, per the original script's rationale.
const fixed_idx_mapk = [54, 55, 56, 57, 58, 62, 63]
const free_idx_mapk = setdiff(1:68, fixed_idx_mapk)
@assert length(free_idx_mapk) == 61

# ---- Bounds for the free parameters: nominal/50 to nominal*50 ----
const bound_factor_low_mapk = 1/50
const bound_factor_high_mapk = 50.0
x0_free_mapk = original_parameters_mapk[free_idx_mapk]
lower_bounds_mapk = x0_free_mapk .* bound_factor_low_mapk
upper_bounds_mapk = x0_free_mapk .* bound_factor_high_mapk
# zero-valued parameters would otherwise get a degenerate [0,0] bound; matches
# the MATLAB zeroMask handling (none of the 61 free params are currently zero,
# but this guards against it if you change the parameterization later)
for i in eachindex(x0_free_mapk)
    if x0_free_mapk[i] <= 0
        lower_bounds_mapk[i] = 0.0
        upper_bounds_mapk[i] = max(1e-9, x0_free_mapk[i] + 1e-9)
    end
end

# ---- Initial conditions (68-element state vector, same order as MATLAB y0) ----
original_u0_mapk = Float64[
    1.0, 0.35, 0.35,            # 1-3:   EGFR
    1.0, 0.245, 0.245,          # 4-6:   Her2
    1.0, 0.203, 0.203,          # 7-9:   Her3
    1.0, 0.0, 1.0,              # 10-12: SHC
    0.0, 0.0,                   # 13-14: Grb2_SOS
    0.0, 0.0,                   # 15-16: HRAS
    0.0, 0.0,                   # 17-18: NRAS
    1.0, 0.0, 1.0,              # 19-21: KRAS
    0.8, 0.366,                 # 22-23: CRAF
    1.0, 1.0,                   # 24-25: BRAF
    1.0, 1.759,                 # 26-27: MEK
    1.0, 1.0,                   # 28-29: ERK (pERK initial set to 1)
    1.0, 2.677,                 # 30-31: DUSP
    1.0, 1.0,                   # 32-33: SPRY
    1.0,                        # 34:    pERK_degrad
    1.0,                        # 35:    pMEK_degrad
    1.0,                        # 36:    pCRAF_degrad
    1.0,                        # 37:    DUSP_stop
    1.0, 0.0, 0.0,              # 38-40: IGFR
    1.0, 0.0,                   # 41-42: IRS
    1.0,                        # 43:    p85
    0.1,                        # 44:    p85_EGFR
    0.1,                        # 45:    p85_Her2
    0.1,                        # 46:    p85_Her3
    0.1,                        # 47:    p85_IGFR
    1.0, 0.2,                   # 48-49: PI3K
    1.0, 0.1,                   # 50-51: PIP
    1.0, 0.513,                 # 52-53: AKT
    0.0,                        # 54:    FOXO
    1.0, 0.5,                   # 55-56: mTORC
    1.0, 0.5, 1.002,            # 57-59: frebp1
    1.0, 0.0,                   # 60-61: KSR
    0.0,                        # 62:    BRAF_CRAF_dimer
    1.0, 0.474, 0.474,          # 63-65: PDGFR
    1.0, 1.432,                 # 66-67: S6K
    0.1,                        # 68:    p85_PDGFR
]
@assert length(original_u0_mapk) == 68 "Expected 68 initial conditions, got $(length(original_u0_mapk))"

# ---- Real experimental data ----
# Time points: 0, 1, 4, 8, 24, 48 hours -> minutes (rate constants above are per-minute;
# the original MATLAB script had a 60x unit bug here -- fixed by using minutes, not seconds)
const timestamps_hours_mapk = [0.0, 1.0, 4.0, 8.0, 24.0, 48.0]
const timestamps_minutes_mapk = timestamps_hours_mapk .* 60.0

const exp_data_raw_mapk = Dict(
    :panRAS => [0.967381946, 1.017734223, 1.0307083,   1.077694732, 1.298017607, 1.573403892],
    :pMEK   => [1.936660577, 0.029380652, 0.012873835, 0.03390921,  0.095155796, 0.944936578],
    :pERK   => [3.273353557, 0.075717978, 0.011570416, 0.00642985,  0.041863585, 0.91621491],
    :DUSP   => [2.854207662, 2.842703936, 1.163746208, 0.332720449, 0.030434242, 0.094073888],
    :pEGFR  => [0.222379739, 0.622877159, 0.629217784, 0.533530834, 0.022513609, 0.010036399],
    :pCRAF  => [0.234376572, 0.641878896, 0.567434544, 0.406320223, 0.582899195, 0.25113447],
    :pAKT   => [0.527301325, 0.614645732, 0.95895017,  0.895019432, 0.412820453, 0.269891704],
    :p4EBP1 => [0.793559668, 1.176099875, 1.210864904, 1.415698564, 0.858543042, 0.167293554],
    :pS6K   => [1.385578651, 1.388228355, 1.286010223, 0.720958901, 0.12299088,  0.028906108],
    :pHer2  => [0.306924546, 0.275751955, 0.32171108,  0.23070312,  1.013023288, 1.045536401],
    :pHer3  => [0.295284147, 0.285719072, 0.385045943, 0.582261781, 0.751301308, 0.264889608],
    :pDGFR  => [0.583361128, 0.585284809, 0.785897279, 1.208147444, 2.298226921, 2.387788835],
)

# Per-species min-max normalization, matching MATLAB's expData_norm
exp_data_norm_mapk = Dict(k => min_max_normalize(v) for (k, v) in exp_data_raw_mapk)

# Per-species residual weights, in the SAME order as OBSERVED_SPECIES above
# [pEGFR, panRAS, pMEK, pERK, DUSP, pCRAF, pAKT, p4EBP1, pHer2, pHer3, pDGFR, pS6K]
const species_weights_mapk = [3.0, 12.0, 10.0, 10.0, 2.0, 10.0, 15.0, 5.0, 5.0, 5.0, 8.0, 4.0]

initial_time_training_mapk = 0.0
end_time_training_mapk = timestamps_minutes_mapk[end]

# ---- HNODE-specific bookkeeping: paradoxical RAF activation (parameter 61,
# kParadoxCRAF) is replaced by a neural network, so it's removed from the set
# of mechanistic parameters being fitted. ----
const nn_replaced_idx_mapk = [61]
const free_idx_hnode_mapk = setdiff(free_idx_mapk, nn_replaced_idx_mapk)
@assert length(free_idx_hnode_mapk) == 60

x0_free_hnode_mapk = original_parameters_mapk[free_idx_hnode_mapk]

# NOTE: the full nominal/50-nominal*50 range (bound_factor_low_mapk/high_mapk above) is
# far too wide to sample independently across 60 parameters simultaneously on a system
# this large and stiff -- the joint probability that all 60 land somewhere numerically
# integrable is essentially zero, which is why Step 2a was hitting the 1e6 fallback loss
# on every single trial. Use a much tighter range for the actual TPE search space. Widen
# this later (e.g. toward 1/2..2) once training is working reliably, if you want more
# aggressive global exploration.
const hnode_tuning_bound_factor_low = 1/1.5
const hnode_tuning_bound_factor_high = 1.5
lower_bounds_hnode_mapk = x0_free_hnode_mapk .* hnode_tuning_bound_factor_low
upper_bounds_hnode_mapk = x0_free_hnode_mapk .* hnode_tuning_bound_factor_high
for i in eachindex(x0_free_hnode_mapk)
    if x0_free_hnode_mapk[i] <= 0
        lower_bounds_hnode_mapk[i] = 0.0
        upper_bounds_hnode_mapk[i] = max(1e-9, x0_free_hnode_mapk[i] + 1e-9)
    end
end

# Known constant values at fixed_idx_mapk (Tram.conc, Ki_RAF, Ki_KSR, Tram.Hill_n,
# Vem.conc, Vem.IC50, Vem.Hill_n), needed to reconstruct the full 68-vector inside
# the HNODE derivative function.
const fixed_values_mapk = original_parameters_mapk[fixed_idx_mapk]