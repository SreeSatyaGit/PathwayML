#=
Settings for the MAPK/PI3K crosstalk model under Vemurafenib (BRAF inhibitor) +
Trametinib (MEK inhibitor) combination treatment.

*** EVERYTHING IN THIS FILE IS A PLACEHOLDER ***
Replace with your own literature/experimentally-derived values before running anything.

State ordering (5 states, all normalized to [0,1] fractional activation):
  u[1] = pRAF  (active RAF)
  u[2] = pMEK  (active MEK)
  u[3] = pERK  (active ERK)
  u[4] = pPI3K (active PI3K)
  u[5] = pAKT  (active AKT)
=#

# ---- Mechanistic parameters (TODO: replace with real values / literature estimates) ----
# order: [k_raf, d_raf, k_mek, d_mek, k_erk, d_erk, k_pi3k, d_pi3k, k_akt, d_akt]
original_parameters = Float64[
    1.0,   # k_raf   - RAS-driven RAF activation rate
    0.5,   # d_raf   - RAF deactivation rate
    1.0,   # k_mek   - RAF -> MEK activation rate
    0.5,   # d_mek   - MEK deactivation rate
    1.0,   # k_erk   - MEK -> ERK activation rate
    0.5,   # d_erk   - ERK deactivation rate
    0.8,   # k_pi3k  - basal PI3K activation rate
    0.4,   # d_pi3k  - PI3K deactivation rate (includes ERK negative feedback below)
    1.0,   # k_akt   - PI3K -> AKT activation rate
    0.5,   # d_akt   - AKT deactivation rate
]

# ---- "Unknown" crosstalk parameter (used ONLY to generate synthetic ground-truth data) ----
# This represents AKT-mediated bypass reactivation of RAF -- the mechanism by which
# tumors partially escape BRAF/MEK inhibition. In the HNODE model this term is replaced
# by a neural network; this ground-truth value is what you're testing the pipeline against.
# TODO: if you have real data instead of synthetic data, delete this and skip the ground-truth model entirely.
crosstalk_akt_raf = 0.6

# ---- Drug pharmacodynamics (TODO: replace with real IC50s / PK if available) ----
IC50_vem  = 0.5   # vemurafenib potency on RAF
IC50_tram = 0.2   # trametinib potency on MEK

# Dosing regimen: constant concentration from t_dose_start to t_dose_end, 0 otherwise.
# TODO: replace with a real PK profile (e.g. a one/two-compartment model) if you have one.
vem_dose  = 1.0
tram_dose = 1.0
t_dose_start = 0.0
t_dose_end   = 24.0

function vem_concentration(t)
    return (t >= t_dose_start && t <= t_dose_end) ? vem_dose : 0.0
end

function tram_concentration(t)
    return (t >= t_dose_start && t <= t_dose_end) ? tram_dose : 0.0
end

# ---- Initial conditions (TODO: replace with real basal activation levels) ----
# All states are normalized fractions in [0, 1]; assume low basal pathway activity pre-treatment.
original_u0 = [0.1, 0.1, 0.1, 0.1, 0.1]

# ---- Training time span (TODO: match your experimental sampling window, e.g. hours) ----
initial_time_training = 0.0f0
end_time_training = 48.0f0
