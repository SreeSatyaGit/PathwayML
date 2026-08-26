#=
Faithful Julia translation of the user's MATLAB `Mapk_ODE` function
(MAPK/PI3K pathway with Vemurafenib + Trametinib combination therapy).

This is a FULLY mechanistic model as provided -- every term has a known kinetic
form and every one of the 68 parameters is either fitted (free) or fixed
(drug dose / IC50 / Ki / Hill coefficients). There is currently no "unknown"
component designated for a neural network to learn -- see the note at the
bottom of this file about how to turn this into an HNODE if you want one.

State indexing matches the MATLAB script exactly (Julia is also 1-indexed,
so no index shifting was needed). Species groupings, observed outputs, and
the per-species weighted residual structure are preserved as-is.
=#

"""
    mapk_ode!(du, u, p, t)

In-place derivative function for the 68-state MAPK/PI3K + Vemurafenib/Trametinib
model. `p` is the full 68-element parameter vector (same order as the MATLAB
`params_vector`).
"""
function mapk_ode!(du, u, p, t)
    ka1=p[1]; kr1=p[2]; kc1=p[3]; kpCraf=p[4]; kpMek=p[5]; kpErk=p[6]; kDegradEgfr=p[7]
    kErkInbEgfr=p[8]; kShcDephos=p[9]; kptpDeg=p[10]; kGrb2CombShc=p[11]; kSprtyInbGrb2=p[12]
    kSosCombGrb2=p[13]; kErkPhosSos=p[14]; kErkPhosPcraf=p[15]; kPcrafDegrad=p[16]
    kErkPhosMek=p[17]; kMekDegrad=p[18]; kDuspInbErk=p[19]; kErkDeg=p[20]; kinbBraf=p[21]
    kDuspStop=p[22]; kDusps=p[23]; kSproutyForm=p[24]; kSprtyComeDown=p[25]; kdegrad=p[26]
    km_Sprty_decay=p[27]; km_Dusp=p[28]; km_Sprty=p[29]; kErkDephos=p[30]; kDuspDeg=p[31]
    kHer2_act=p[32]; kHer3_act=p[33]; k_p85_bind_EGFR=p[34]; k_p85_bind_Her2=p[35]
    k_p85_bind_Her3=p[36]; k_p85_bind_IGFR=p[37]; k_p85_unbind=p[38]; k_PI3K_recruit=p[39]
    kMTOR_Feedback=p[40]; k_PIP2_to_PIP3=p[41]; k_PTEN=p[42]; kAkt=p[43]; kdegradAKT=p[44]
    kb1=p[45]; k43b1=p[46]; k4ebp1=p[47]; k_4EBP1_dephos=p[48]; kKSRphos=p[49]; kKSRdephos=p[50]
    kMekByBraf=p[51]; kMekByCraf=p[52]; kMekByKSR=p[53]
    Tram=p[54]; K_tram_RAF=p[55]; K_tram_KSR=p[56]; n_tram=p[57]
    Vemurafenib=p[58]; kDimerForm=p[59]; kDimerDissoc=p[60]; kParadoxCRAF=p[61]
    IC50_vem=p[62]; Hill_n_vem=p[63]
    kPDGFR_act=p[64]; k_p85_bind_PDGFR=p[65]; kS6K_phos=p[66]; kS6K_dephos=p[67]
    K_displace=p[68]

    du[1] = -ka1*u[1]*u[2]
    du[2] = -ka1*u[1]*u[2] - kr1*u[2] - kc1*u[2]
    du[3] =  kc1*u[2] - kDegradEgfr*u[3] - kErkInbEgfr*u[29]*u[3]
    du[4] = -kHer2_act*u[4] + kr1*u[5]
    du[5] = kHer2_act*u[4] - kr1*u[5] - kc1*u[5]
    du[6] = kc1*u[5] - kDegradEgfr*u[6] - kErkInbEgfr*u[29]*u[6]
    du[7] = -kHer3_act*u[7] + kr1*u[8]
    du[8] = kHer3_act*u[7] - kr1*u[8] - kc1*u[8]
    du[9] = kc1*u[8] - kDegradEgfr*u[9] - kErkInbEgfr*u[29]*u[9]

    du[10] = -ka1*u[3]*u[10]
    du[11] = ka1*u[3]*u[10] - kShcDephos*u[12]*u[11]
    du[12] = -kptpDeg*u[11]*u[12]
    du[13] = kGrb2CombShc*u[11]*u[3] - kSprtyInbGrb2*u[33]*u[13]
    du[14] = kSosCombGrb2*u[13]*u[11] - kErkPhosSos*u[29]*u[14]

    du[15] = -ka1*u[14]*u[15] + kdegrad*u[16]
    du[16] =  ka1*u[14]*u[15] - kdegrad*u[16]
    du[17] = -ka1*u[14]*u[17] + kdegrad*u[18]
    du[18] =  ka1*u[14]*u[17] - kdegrad*u[18]
    du[19] = -ka1*u[14]*u[19] + kdegrad*u[21]
    du[20] = 0.0
    du[21] =  ka1*u[14]*u[19] - kdegrad*u[21]
    panRAS_active = u[16] + u[18] + u[21]

    IC50_n = IC50_vem^Hill_n_vem
    Vem_n = Vemurafenib^Hill_n_vem
    kBRAF_eff = ka1 * IC50_n / (IC50_n + Vem_n + eps())
    paradox_activation = kParadoxCRAF * Vemurafenib * u[62]

    du[22] = -kpCraf*panRAS_active*u[22] + kErkPhosPcraf*u[29]*u[23] + kPcrafDegrad*u[23]*u[36] - kDimerForm*u[25]*u[22]*Vemurafenib + kDimerDissoc*u[62]
    du[23] = kpCraf*panRAS_active*u[22] - kErkPhosPcraf*u[29]*u[23] - kPcrafDegrad*u[23]*u[36] + paradox_activation
    du[24] = -kBRAF_eff*u[24]*panRAS_active - kDimerForm*u[25]*u[22]*Vemurafenib + kDimerDissoc*u[62]
    du[25] = kBRAF_eff*u[24]*panRAS_active - kinbBraf*u[25] - kDimerForm*u[25]*u[22]*Vemurafenib + kDimerDissoc*u[62]
    du[62] = kDimerForm*u[25]*u[22]*Vemurafenib - kDimerDissoc*u[62] - kPcrafDegrad*u[62]*u[36]

    Stream_Load = panRAS_active + (u[22]+u[23]) + (u[24]+u[25])
    Ki_effective = K_tram_RAF * (1 + (Stream_Load / K_displace)^2)
    f_MEK_activity = 1 / (max(0.01, 1 + (Tram / max(eps(), Ki_effective))^n_tram))

    raf_to_mek = (kpMek*u[23] + kMekByBraf*u[25] + kMekByCraf*u[23] + kpMek*u[62])
    ksr_to_mek = (kMekByKSR * u[61])
    du[26] = -(raf_to_mek + ksr_to_mek)*u[26] + kErkPhosMek*u[29]*u[27] + kMekDegrad*u[27]*u[35]
    du[27] = (raf_to_mek + ksr_to_mek)*u[26] - kErkPhosMek*u[29]*u[27] - kMekDegrad*u[27]*u[35]

    erk_activation = kpErk * u[27] * u[28] * f_MEK_activity
    du[28] = -erk_activation + kErkDephos*u[31]*u[29] + kErkDeg*u[29]*u[34]
    du[29] = erk_activation - kErkDephos*u[31]*u[29] - kErkDeg*u[29]*u[34]

    du[30] = km_Dusp*u[29]/(1 + (km_Dusp/kDusps)*u[29]) - kDuspStop*u[30]*u[37] - kDuspDeg*u[30]*u[29]
    du[31] = -kDuspStop*u[30]*u[31]
    du[32] = km_Sprty*u[29]/(1 + (km_Sprty/kSproutyForm)*u[29]) - kSprtyComeDown*u[32]*u[33]
    du[33] = -kSprtyComeDown*u[32]*u[33]
    du[34] = -kErkDeg*u[29]*u[34]
    du[35] = -kMekDegrad*u[27]*u[35]
    du[36] = -kPcrafDegrad*u[23]*u[36]
    du[37] = -kDuspStop*u[30]*u[37]
    du[38] = -ka1*u[38] + kr1*u[39]
    du[39] = ka1*u[38] - kr1*u[39] - kc1*u[39]
    du[40] = kc1*u[39] - kErkInbEgfr*u[29]*u[40]
    du[41] = -ka1*u[3]*u[41]
    du[42] = ka1*u[3]*u[41]
    du[43] = 0.0  # total free p85 pool -- never explicitly assigned in the original MATLAB
                  # (dydt(43) is left at its zeros(68,1) default), i.e. treated as a constant
                  # reservoir. Preserved here exactly as in the original.
    du[44] = k_p85_bind_EGFR*u[3]*u[43] - k_p85_unbind*u[44]
    du[45] = k_p85_bind_Her2*u[6]*u[43] - k_p85_unbind*u[45]
    du[46] = k_p85_bind_Her3*u[9]*u[43] - k_p85_unbind*u[46]
    du[47] = k_p85_bind_IGFR*u[40]*u[43] - k_p85_unbind*u[47]
    du[68] = k_p85_bind_PDGFR*u[65]*u[43] - k_p85_unbind*u[68]
    total_p85_RTK = u[44] + u[45] + u[46] + u[47] + u[68]
    du[48] = -k_PI3K_recruit*total_p85_RTK*u[48] + kMTOR_Feedback*u[56]*u[49]
    du[49] = k_PI3K_recruit*total_p85_RTK*u[48] - kMTOR_Feedback*u[56]*u[49]
    du[50] = -k_PIP2_to_PIP3*u[49]*u[50] + k_PTEN*u[51]
    du[51] = k_PIP2_to_PIP3*u[49]*u[50] - k_PTEN*u[51]
    du[52] = -kAkt*u[51]*u[52] + kdegradAKT*u[53]
    du[53] = kAkt*u[51]*u[52] - kdegradAKT*u[53]
    du[54] = (max(0, 1 - u[53])) * kAkt / (max(0.1, 1 + (u[54] / 15e-5)))
    du[55] = -kAkt * u[53] * u[55] + kdegrad * u[56]
    du[56] = kAkt * u[53] * u[55] - kdegrad * u[56]
    du[57] = -k4ebp1*u[56]*u[57] + kb1*u[58] + k_4EBP1_dephos*u[59]
    du[58] = k4ebp1*u[56]*u[57] - kb1*u[58] - k43b1*u[58]
    du[59] = k43b1*u[58] - k_4EBP1_dephos*u[59]
    du[60] = -kKSRphos*panRAS_active*u[60] + kKSRdephos*u[61]
    du[61] = kKSRphos*panRAS_active*u[60] - kKSRdephos*u[61]
    du[63] = -kPDGFR_act*u[63] + kr1*u[64]
    du[64] = kPDGFR_act*u[63] - kr1*u[64] - kc1*u[64]
    du[65] = kc1*u[64] - kDegradEgfr*u[65] - kErkInbEgfr*u[29]*u[65]
    du[66] = -kS6K_phos*u[56]*u[66] + kS6K_dephos*u[67]
    du[67] = kS6K_phos*u[56]*u[66] - kS6K_dephos*u[67]
end

"""
Indices (into the 68-state vector) of the 12 experimentally observed species,
and how to compute each from the raw state (some are sums of sub-states, e.g.
panRAS = HRAS_active + NRAS_active + KRAS_active).
"""
const OBSERVED_SPECIES = (
    pEGFR  = u -> u[3],
    panRAS = u -> u[16] + u[18] + u[21],
    pCRAF  = u -> u[23],
    pMEK   = u -> u[27],
    pERK   = u -> u[29],
    DUSP   = u -> u[31],
    pAKT   = u -> u[53],
    p4EBP1 = u -> u[59],
    pHer2  = u -> u[6],
    pHer3  = u -> u[9],
    pDGFR  = u -> u[65],
    pS6K   = u -> u[67],
)

"""
    min_max_normalize(v)

Matches MATLAB's `(v - min(v)) ./ (max(v) - min(v) + eps)` used throughout the
original script. Note this normalizes ACROSS the whole trajectory (a global,
not pointwise, operation) -- see the note in the settings file about what this
means for how the loss function must be structured.
"""
min_max_normalize(v) = (v .- minimum(v)) ./ (maximum(v) - minimum(v) + eps())

#=
NOTE ON TURNING THIS INTO AN HNODE:

This model, as given, is fully mechanistic -- there is no term here that is
"unknown" in the HNODE sense (i.e. no term you don't have a kinetic law for
and want a neural network to learn instead). Before this can go through Step 2a
(hyperparameter tuning) the way the rest of this scaffold's pipeline expects,
you need to decide one of:

  (a) Skip the neural network entirely and treat this as a pure mechanistic
      parameter-estimation + identifiability-analysis problem -- i.e. use
      Steps 3-4's Hessian/FIM machinery directly on the 61 free parameters,
      with your own gradient-based fit (much closer to what MATLAB's
      lsqnonlin was already doing, but adding the identifiability analysis
      MATLAB doesn't give you), OR

  (b) Pick a SPECIFIC term/interaction in this model you're least confident
      about the functional form of (e.g. the paradox activation term, the
      Tram/Ki_effective displacement kinetics, or the PI3K/mTOR feedback loop)
      and replace just that term with `NN(relevant_inputs, p.p_net, st)`,
      the same way the illustrative vem_tram scaffold did for the AKT->RAF
      crosstalk term.

Which one applies changes how Steps 2a/2b need to be rewritten around this
model, so this file intentionally stops here until that's decided.
=#
