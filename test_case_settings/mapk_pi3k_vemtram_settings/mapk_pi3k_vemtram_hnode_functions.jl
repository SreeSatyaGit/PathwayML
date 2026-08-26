#=
HNODE variant of the 68-state MAPK/PI3K + Vem/Tram model. Every equation is
identical to `mapk_ode!` in mapk_pi3k_vemtram_model_functions.jl EXCEPT for one
line: the paradoxical RAF activation term

    paradox_activation = kParadoxCRAF * Vemurafenib * u[62]

(previously added into du[23], the pCRAF equation) is replaced by

    paradox_activation = NN([u[62], Vemurafenib], p.p_net, st)[1]

i.e. the neural network learns the entire functional relationship between the
BRAF-CRAF dimer level, the Vemurafenib concentration, and the resulting
paradoxical CRAF activation rate -- not just a scalar rate constant. This
means parameter 61 (kParadoxCRAF) is no longer a mechanistic parameter to fit;
it's dropped from the free-parameter set (see `free_idx_hnode` in the
settings file).

Because Vemurafenib concentration is constant within any one simulated
trajectory in this dataset (no time-varying dosing/PK model here), the NN
effectively learns a 1-D function of dimer level for this particular dataset.
Passing the concentration as an explicit second input anyway preserves the
correct causal structure and means this would generalize correctly if you
later fit against a dose-ranging dataset instead of a single fixed dose.
=#

"""
    assemble_full_parameter_vector(mechanistic_scaled, free_idx_no_nn, fixed_idx, fixed_values)

Reconstructs the full 68-length parameter vector needed by the equations from:
  - the free (fitted) mechanistic parameters, scaled by their original nominal
    values, placed at `free_idx_no_nn`
  - the known constants placed at `fixed_idx`
  - position 61 (kParadoxCRAF) is left at 0 and is simply unused, since the
    paradox activation term is computed by the neural network instead.
"""
function assemble_full_parameter_vector(mechanistic_scaled::AbstractVector,
                                          free_idx_no_nn::Vector{Int},
                                          fixed_idx::Vector{Int},
                                          fixed_values::Vector{Float64})
    p_full = zeros(eltype(mechanistic_scaled), 68)
    p_full[free_idx_no_nn] .= mechanistic_scaled
    p_full[fixed_idx] .= fixed_values
    return p_full
end

"""
    get_uode_model_function(appr_neural_network, state, original_parameters_opt, free_idx_no_nn, fixed_idx, fixed_values)

Returns the HNODE derivative function `f(du, u, p, t)` where `p` is expected to
be a ComponentArray with `p.ode_par` (60 free mechanistic parameters, scaled to
~1.0) and `p.p_net` (neural network parameters).
"""
function get_uode_model_function(appr_neural_network, state, original_parameters_opt,
                                   free_idx_no_nn, fixed_idx, fixed_values)
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state,
            original_parameters_opt = original_parameters_opt,
            free_idx_no_nn = free_idx_no_nn, fixed_idx = fixed_idx, fixed_values = fixed_values

            mechanistic_scaled = p.ode_par .* original_parameters_opt
            p_full = assemble_full_parameter_vector(mechanistic_scaled, free_idx_no_nn, fixed_idx, fixed_values)

            ka1=p_full[1]; kr1=p_full[2]; kc1=p_full[3]; kpCraf=p_full[4]; kpMek=p_full[5]; kpErk=p_full[6]; kDegradEgfr=p_full[7]
            kErkInbEgfr=p_full[8]; kShcDephos=p_full[9]; kptpDeg=p_full[10]; kGrb2CombShc=p_full[11]; kSprtyInbGrb2=p_full[12]
            kSosCombGrb2=p_full[13]; kErkPhosSos=p_full[14]; kErkPhosPcraf=p_full[15]; kPcrafDegrad=p_full[16]
            kErkPhosMek=p_full[17]; kMekDegrad=p_full[18]; kDuspInbErk=p_full[19]; kErkDeg=p_full[20]; kinbBraf=p_full[21]
            kDuspStop=p_full[22]; kDusps=p_full[23]; kSproutyForm=p_full[24]; kSprtyComeDown=p_full[25]; kdegrad=p_full[26]
            km_Sprty_decay=p_full[27]; km_Dusp=p_full[28]; km_Sprty=p_full[29]; kErkDephos=p_full[30]; kDuspDeg=p_full[31]
            kHer2_act=p_full[32]; kHer3_act=p_full[33]; k_p85_bind_EGFR=p_full[34]; k_p85_bind_Her2=p_full[35]
            k_p85_bind_Her3=p_full[36]; k_p85_bind_IGFR=p_full[37]; k_p85_unbind=p_full[38]; k_PI3K_recruit=p_full[39]
            kMTOR_Feedback=p_full[40]; k_PIP2_to_PIP3=p_full[41]; k_PTEN=p_full[42]; kAkt=p_full[43]; kdegradAKT=p_full[44]
            kb1=p_full[45]; k43b1=p_full[46]; k4ebp1=p_full[47]; k_4EBP1_dephos=p_full[48]; kKSRphos=p_full[49]; kKSRdephos=p_full[50]
            kMekByBraf=p_full[51]; kMekByCraf=p_full[52]; kMekByKSR=p_full[53]
            Tram=p_full[54]; K_tram_RAF=p_full[55]; K_tram_KSR=p_full[56]; n_tram=p_full[57]
            Vemurafenib=p_full[58]
            kDimerForm=p_full[59]; kDimerDissoc=p_full[60]
            # p_full[61] (kParadoxCRAF) intentionally unused -- the NN replaces this term
            IC50_vem=p_full[62]; Hill_n_vem=p_full[63]
            kPDGFR_act=p_full[64]; k_p85_bind_PDGFR=p_full[65]; kS6K_phos=p_full[66]; kS6K_dephos=p_full[67]
            K_displace=p_full[68]

            @inbounds du[1] = -ka1*u[1]*u[2]
            @inbounds du[2] = -ka1*u[1]*u[2] - kr1*u[2] - kc1*u[2]
            @inbounds du[3] =  kc1*u[2] - kDegradEgfr*u[3] - kErkInbEgfr*u[29]*u[3]
            @inbounds du[4] = -kHer2_act*u[4] + kr1*u[5]
            @inbounds du[5] = kHer2_act*u[4] - kr1*u[5] - kc1*u[5]
            @inbounds du[6] = kc1*u[5] - kDegradEgfr*u[6] - kErkInbEgfr*u[29]*u[6]
            @inbounds du[7] = -kHer3_act*u[7] + kr1*u[8]
            @inbounds du[8] = kHer3_act*u[7] - kr1*u[8] - kc1*u[8]
            @inbounds du[9] = kc1*u[8] - kDegradEgfr*u[9] - kErkInbEgfr*u[29]*u[9]

            @inbounds du[10] = -ka1*u[3]*u[10]
            @inbounds du[11] = ka1*u[3]*u[10] - kShcDephos*u[12]*u[11]
            @inbounds du[12] = -kptpDeg*u[11]*u[12]
            @inbounds du[13] = kGrb2CombShc*u[11]*u[3] - kSprtyInbGrb2*u[33]*u[13]
            @inbounds du[14] = kSosCombGrb2*u[13]*u[11] - kErkPhosSos*u[29]*u[14]

            @inbounds du[15] = -ka1*u[14]*u[15] + kdegrad*u[16]
            @inbounds du[16] =  ka1*u[14]*u[15] - kdegrad*u[16]
            @inbounds du[17] = -ka1*u[14]*u[17] + kdegrad*u[18]
            @inbounds du[18] =  ka1*u[14]*u[17] - kdegrad*u[18]
            @inbounds du[19] = -ka1*u[14]*u[19] + kdegrad*u[21]
            @inbounds du[20] = 0.0
            @inbounds du[21] =  ka1*u[14]*u[19] - kdegrad*u[21]
            panRAS_active = u[16] + u[18] + u[21]

            IC50_n = IC50_vem^Hill_n_vem
            Vem_n = Vemurafenib^Hill_n_vem
            kBRAF_eff = ka1 * IC50_n / (IC50_n + Vem_n + eps())

            # === THE REPLACED TERM ===
            # Original: paradox_activation = kParadoxCRAF * Vemurafenib * u[62]
            nn_input = [u[62], Vemurafenib]
            û = appr_neural_network(nn_input, p.p_net, st)[1]
            paradox_activation = û[1]
            # ==========================

            @inbounds du[22] = -kpCraf*panRAS_active*u[22] + kErkPhosPcraf*u[29]*u[23] + kPcrafDegrad*u[23]*u[36] - kDimerForm*u[25]*u[22]*Vemurafenib + kDimerDissoc*u[62]
            @inbounds du[23] = kpCraf*panRAS_active*u[22] - kErkPhosPcraf*u[29]*u[23] - kPcrafDegrad*u[23]*u[36] + paradox_activation
            @inbounds du[24] = -kBRAF_eff*u[24]*panRAS_active - kDimerForm*u[25]*u[22]*Vemurafenib + kDimerDissoc*u[62]
            @inbounds du[25] = kBRAF_eff*u[24]*panRAS_active - kinbBraf*u[25] - kDimerForm*u[25]*u[22]*Vemurafenib + kDimerDissoc*u[62]
            @inbounds du[62] = kDimerForm*u[25]*u[22]*Vemurafenib - kDimerDissoc*u[62] - kPcrafDegrad*u[62]*u[36]

            Stream_Load = panRAS_active + (u[22]+u[23]) + (u[24]+u[25])
            Ki_effective = K_tram_RAF * (1 + (Stream_Load / K_displace)^2)
            f_MEK_activity = 1 / (max(0.01, 1 + (Tram / max(eps(), Ki_effective))^n_tram))

            raf_to_mek = (kpMek*u[23] + kMekByBraf*u[25] + kMekByCraf*u[23] + kpMek*u[62])
            ksr_to_mek = (kMekByKSR * u[61])
            @inbounds du[26] = -(raf_to_mek + ksr_to_mek)*u[26] + kErkPhosMek*u[29]*u[27] + kMekDegrad*u[27]*u[35]
            @inbounds du[27] = (raf_to_mek + ksr_to_mek)*u[26] - kErkPhosMek*u[29]*u[27] - kMekDegrad*u[27]*u[35]

            erk_activation = kpErk * u[27] * u[28] * f_MEK_activity
            @inbounds du[28] = -erk_activation + kErkDephos*u[31]*u[29] + kErkDeg*u[29]*u[34]
            @inbounds du[29] = erk_activation - kErkDephos*u[31]*u[29] - kErkDeg*u[29]*u[34]

            @inbounds du[30] = km_Dusp*u[29]/(1 + (km_Dusp/kDusps)*u[29]) - kDuspStop*u[30]*u[37] - kDuspDeg*u[30]*u[29]
            @inbounds du[31] = -kDuspStop*u[30]*u[31]
            @inbounds du[32] = km_Sprty*u[29]/(1 + (km_Sprty/kSproutyForm)*u[29]) - kSprtyComeDown*u[32]*u[33]
            @inbounds du[33] = -kSprtyComeDown*u[32]*u[33]
            @inbounds du[34] = -kErkDeg*u[29]*u[34]
            @inbounds du[35] = -kMekDegrad*u[27]*u[35]
            @inbounds du[36] = -kPcrafDegrad*u[23]*u[36]
            @inbounds du[37] = -kDuspStop*u[30]*u[37]
            @inbounds du[38] = -ka1*u[38] + kr1*u[39]
            @inbounds du[39] = ka1*u[38] - kr1*u[39] - kc1*u[39]
            @inbounds du[40] = kc1*u[39] - kErkInbEgfr*u[29]*u[40]
            @inbounds du[41] = -ka1*u[3]*u[41]
            @inbounds du[42] = ka1*u[3]*u[41]
            @inbounds du[43] = 0.0
            @inbounds du[44] = k_p85_bind_EGFR*u[3]*u[43] - k_p85_unbind*u[44]
            @inbounds du[45] = k_p85_bind_Her2*u[6]*u[43] - k_p85_unbind*u[45]
            @inbounds du[46] = k_p85_bind_Her3*u[9]*u[43] - k_p85_unbind*u[46]
            @inbounds du[47] = k_p85_bind_IGFR*u[40]*u[43] - k_p85_unbind*u[47]
            @inbounds du[68] = k_p85_bind_PDGFR*u[65]*u[43] - k_p85_unbind*u[68]
            total_p85_RTK = u[44] + u[45] + u[46] + u[47] + u[68]
            @inbounds du[48] = -k_PI3K_recruit*total_p85_RTK*u[48] + kMTOR_Feedback*u[56]*u[49]
            @inbounds du[49] = k_PI3K_recruit*total_p85_RTK*u[48] - kMTOR_Feedback*u[56]*u[49]
            @inbounds du[50] = -k_PIP2_to_PIP3*u[49]*u[50] + k_PTEN*u[51]
            @inbounds du[51] = k_PIP2_to_PIP3*u[49]*u[50] - k_PTEN*u[51]
            @inbounds du[52] = -kAkt*u[51]*u[52] + kdegradAKT*u[53]
            @inbounds du[53] = kAkt*u[51]*u[52] - kdegradAKT*u[53]
            @inbounds du[54] = (max(0, 1 - u[53])) * kAkt / (max(0.1, 1 + (u[54] / 15e-5)))
            @inbounds du[55] = -kAkt * u[53] * u[55] + kdegrad * u[56]
            @inbounds du[56] = kAkt * u[53] * u[55] - kdegrad * u[56]
            @inbounds du[57] = -k4ebp1*u[56]*u[57] + kb1*u[58] + k_4EBP1_dephos*u[59]
            @inbounds du[58] = k4ebp1*u[56]*u[57] - kb1*u[58] - k43b1*u[58]
            @inbounds du[59] = k43b1*u[58] - k_4EBP1_dephos*u[59]
            @inbounds du[60] = -kKSRphos*panRAS_active*u[60] + kKSRdephos*u[61]
            @inbounds du[61] = kKSRphos*panRAS_active*u[60] - kKSRdephos*u[61]
            @inbounds du[63] = -kPDGFR_act*u[63] + kr1*u[64]
            @inbounds du[64] = kPDGFR_act*u[63] - kr1*u[64] - kc1*u[64]
            @inbounds du[65] = kc1*u[64] - kDegradEgfr*u[65] - kErkInbEgfr*u[29]*u[65]
            @inbounds du[66] = -kS6K_phos*u[56]*u[66] + kS6K_dephos*u[67]
            @inbounds du[67] = kS6K_phos*u[56]*u[66] - kS6K_dephos*u[67]
        end
end
