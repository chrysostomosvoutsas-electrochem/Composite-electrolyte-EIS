import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress
from scipy.linalg import toeplitz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re

# ==============================================================================
# 1. APPLICATION CONFIGURATION & METADATA
# ==============================================================================

st.set_page_config(
    page_title="Universal Solid-State Blocking Cell EIS Diagnostic Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for high-contrast scientific dashboards
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .metric-header {
        font-size: 0.80rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.40rem;
        color: #0f172a;
        font-weight: 800;
        margin-top: 4px;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0px 0px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. SCALE-INVARIANT PHYSICAL & EQUIVALENT CIRCUIT ENGINE
# ==============================================================================

def model_Z_relaxis(params, w):
    """
    Standard Symmetric Blocking Cell Model (RelaxIS topology):
    Z(w) = R1 + (R2 / (1 + j*w*R2*C1)) + 1 / (Q_dl * (j*w)^alpha_dl)
    """
    R1, R2, C1, Q_dl, a_dl = params
    
    # HF Interfacial/Grain Boundary Arc
    jw = 1j * w
    Z_arc = R2 / (1.0 + jw * R2 * C1)
    
    # Low-Frequency Blocking Double-Layer CPE
    jw_dl = jw ** a_dl
    Z_dl = 1.0 / (Q_dl * jw_dl)
    
    return R1 + Z_arc + Z_dl

def residuals_relaxis(params, w, z_re, z_im):
    """
    Modulus-weighted complex residuals. Normalizing by |Z_exp| ensures
    numerical stability across wide frequency sweeps (mHz to MHz).
    """
    Z_pred = model_Z_relaxis(params, w)
    res_re = (Z_pred.real - z_re) / np.abs(z_re)
    res_im = (-Z_pred.imag - z_im) / np.abs(z_im)
    return np.concatenate([res_re, res_im])

def fit_universal_blocking_spectrum(f, z_re, z_im):
    """
    Dynamic, scale-invariant fitting algorithm. Automatically determines initial
    parameter bounds and guesses from raw data geometry (works from 0.01 Ohm to 10 MOhm).
    """
    w = 2.0 * np.pi * f
    
    # 1. Estimate Bulk Resistance (R1) from high-frequency real intercept
    r1_guess = float(np.min(z_re))
    
    # 2. Estimate HF Semicircle (R2, C1)
    hf_mask = f > np.median(f)
    if np.sum(hf_mask) > 3:
        apex_idx_hf = np.argmax(z_im[hf_mask])
        f_apex = f[hf_mask][apex_idx_hf]
        tau_guess = 1.0 / (2.0 * np.pi * max(f_apex, 1.0))
        r2_guess = max(float(z_re[hf_mask][apex_idx_hf] - r1_guess) * 2.0, r1_guess * 0.2)
    else:
        tau_guess = 1.0 / (2.0 * np.pi * np.max(f) * 0.1)
        r2_guess = r1_guess * 0.5
        
    c1_guess = tau_guess / max(r2_guess, 1e-4)
    
    # 3. Estimate Blocking CPE (Q_dl, alpha) from low-frequency tail
    lf_idx = np.argmin(f)
    z_lf_mag = np.sqrt(z_re[lf_idx]**2 + z_im[lf_idx]**2)
    w_lf = max(w[lf_idx], 1e-6)
    q_dl_guess = float(1.0 / (z_lf_mag * (w_lf ** 0.85)))
    
    # Dynamic order-of-magnitude bounds
    r_min_bound = max(r1_guess * 1e-4, 1e-5)
    r_max_bound = max(np.max(z_re) * 20.0, 1e7)
    
    p0 = [r1_guess, r2_guess, c1_guess, q_dl_guess, 0.85]
    bounds = (
        [r_min_bound, r_min_bound, 1e-16, 1e-14, 0.35],
        [r_max_bound, r_max_bound, 1.0,   10.0,  1.00]
    )
    
    try:
        res = least_squares(
            residuals_relaxis, p0, bounds=bounds,
            args=(w, z_re, z_im), ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=2500
        )
        p = res.x
        
        f_dense = np.logspace(np.log10(np.min(f)), np.log10(np.max(f)), 300)
        w_dense = 2.0 * np.pi * f_dense
        Z_dense = model_Z_relaxis(p, w_dense)
        
        tau = p[1] * p[2]
        f_relax = 1.0 / (2.0 * np.pi * tau) if tau > 0 else 0
        
        Z_fit_pts = model_Z_relaxis(p, w)
        chi_sq = np.sum(((Z_fit_pts.real - z_re)**2 + (-Z_fit_pts.imag - z_im)**2) / (z_re**2 + z_im**2))
        
        return {
            "success": True,
            "R1_bulk": p[0],
            "R2_hf": p[1],
            "R_total": p[0] + p[1],
            "C1": p[2],
            "tau_s": tau,
            "f_relax_kHz": f_relax / 1e3,
            "Q_dl": p[3],
            "alpha_dl": p[4],
            "chi_sq": chi_sq,
            "f_dense": f_dense,
            "Z_re_fit": Z_dense.real,
            "Z_im_fit": -Z_dense.imag,
            "Z_mag_fit": np.abs(Z_dense),
            "phase_fit": np.angle(Z_dense, deg=True)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==============================================================================
# 3. KRAMERS-KRONIG (LIN-KK) RESIDUAL AUDIT
# ==============================================================================

def compute_lin_kk_residuals(f, z_re, z_im, M=None):
    """
    Evaluates physical causality, linearity, and stationarity using a generalized Voigt model.
    """
    N = len(f)
    if M is None:
        M = max(int(N * 0.75), 10)
    
    w = 2.0 * np.pi * f
    w_min, w_max = np.min(w), np.max(w)
    tau_k = np.logspace(-np.log10(w_max), -np.log10(w_min), M)
    
    A_re = np.zeros((N, M + 1))
    A_im = np.zeros((N, M))
    A_re[:, 0] = 1.0
    
    for k in range(M):
        denom = 1.0 + (w * tau_k[k])**2
        A_re[:, k + 1] = 1.0 / denom
        A_im[:, k] = -(w * tau_k[k]) / denom
        
    A = np.vstack([A_re, np.hstack([np.zeros((N, 1)), A_im])])
    b = np.concatenate([z_re, -z_im])
    
    gamma = 1e-4 * np.trace(A.T @ A) / (M + 1)
    coeffs = np.linalg.solve(A.T @ A + gamma * np.eye(M + 1), A.T @ b)
    
    pred = A @ coeffs
    pred_re = pred[:N]
    pred_im = -pred[N:]
    
    delta_re = (z_re - pred_re) / np.abs(z_re)
    delta_im = (z_im - pred_im) / np.abs(z_im)
    
    return {
        "delta_re_pct": delta_re * 100.0,
        "delta_im_pct": delta_im * 100.0,
        "max_residual_pct": max(np.max(np.abs(delta_re)), np.max(np.abs(delta_im))) * 100.0
    }


# ==============================================================================
# 4. REGULARIZED DISTRIBUTION OF RELAXATION TIMES (DRT)
# ==============================================================================

def compute_drt(f, z_re, z_im, lambda_reg=1e-2):
    """
    Continuous distribution of relaxation times gamma(tau) deconvolution.
    """
    w = 2.0 * np.pi * f
    N = len(f)
    tau_min = 1.0 / np.max(w) / 5.0
    tau_max = 1.0 / np.min(w) * 5.0
    M = 100
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), M)
    
    ln_tau = np.log(tau)
    delta_ln_tau = ln_tau[1] - ln_tau[0]
    
    A = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            wt = w[i] * tau[j]
            A[i, j] = wt / (1.0 + wt**2) * delta_ln_tau
            
    b = z_im
    L = np.zeros((M - 1, M))
    for i in range(M - 1):
        L[i, i] = -1.0
        L[i, i + 1] = 1.0
        
    H = A.T @ A + lambda_reg * (L.T @ L)
    g = A.T @ b
    
    try:
        from scipy.optimize import nnls
        gamma, _ = nnls(H, g)
    except:
        gamma = np.linalg.solve(H + 1e-5 * np.eye(M), g)
        gamma = np.clip(gamma, 0, None)
        
    return {
        "tau": tau,
        "freq_equiv": 1.0 / (2.0 * np.pi * tau),
        "gamma": gamma
    }


# ==============================================================================
# 5. VECTOR ARCHITECTURAL SCHEMATICS (SVG RENDERERS)
# ==============================================================================

def render_cell_schematic_svg(thickness_um, diameter_mm, r_bulk=None, r_hf=None):
    r_b_text = f"R_bulk: {r_bulk:.2f} Ω" if r_bulk else "R_bulk (Matrix)"
    r_hf_text = f"R_hf: {r_hf:.2f} Ω" if r_hf else "R_hf (Interphase)"
    
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 420" width="100%" height="320" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <defs>
            <linearGradient id="steelGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#94a3b8" />
                <stop offset="50%" stop-color="#cbd5e1" />
                <stop offset="100%" stop-color="#64748b" />
            </linearGradient>
            <linearGradient id="polyGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#ecfdf5" /><stop offset="50%" stop-color="#d1fae5" /><stop offset="100%" stop-color="#ecfdf5" />
            </linearGradient>
            <radialGradient id="ceramicGrad" cx="35%" cy="35%" r="65%">
                <stop offset="0%" stop-color="#fed7aa" /><stop offset="50%" stop-color="#f97316" /><stop offset="100%" stop-color="#c2410c" />
            </radialGradient>
        </defs>

        <text x="30" y="32" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="14" font-weight="700" fill="#0f172a">
            SYMMETRIC ION-BLOCKING CELL (STAINLESS STEEL | SOLID ELECTROLYTE | STAINLESS STEEL)
        </text>
        <text x="30" y="50" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="11" fill="#64748b">
            Active Area: {np.pi*((diameter_mm/2)*0.1)**2:.3f} cm² | Pellet Thickness: {thickness_um:.1f} µm | Pure Ionic Blocking (t_e ≈ 0)
        </text>

        <!-- TOP CURRENT COLLECTOR -->
        <rect x="70" y="70" width="460" height="45" rx="4" fill="url(#steelGrad)" stroke="#475569" stroke-width="1.5" />
        <text x="210" y="98" font-family="-apple-system,BlinkMacSystemFont" font-size="12" font-weight="700" fill="#0f172a">
            UPPER STAINLESS STEEL ELECTRODE (BLOCKING)
        </text>
        <line x1="300" y1="70" x2="300" y2="40" stroke="#0f172a" stroke-width="3" />
        <circle cx="300" cy="38" r="4" fill="#0284c7" />

        <!-- UPPER DOUBLE LAYER -->
        <rect x="70" y="115" width="460" height="12" fill="#fef9c3" stroke="#eab308" stroke-dasharray="3,3" stroke-width="1" />
        <g fill="#b45309" font-family="monospace" font-size="10" font-weight="bold">
            <text x="90" y="125">Li⁺</text><text x="160" y="125">Li⁺</text><text x="230" y="125">Li⁺</text>
            <text x="300" y="125">Li⁺</text><text x="370" y="125">Li⁺</text><text x="440" y="125">Li⁺</text>
        </g>

        <!-- ELECTROLYTE MEMBRANE -->
        <rect x="70" y="127" width="460" height="150" fill="url(#polyGrad)" stroke="#10b981" stroke-width="2" />
        
        <!-- CERAMIC PARTICLES -->
        <circle cx="130" cy="160" r="18" fill="url(#ceramicGrad)" stroke="#9a3412" stroke-width="1" />
        <circle cx="230" cy="170" r="22" fill="url(#ceramicGrad)" stroke="#9a3412" stroke-width="1" />
        <circle cx="340" cy="155" r="19" fill="url(#ceramicGrad)" stroke="#9a3412" stroke-width="1" />
        <circle cx="450" cy="165" r="21" fill="url(#ceramicGrad)" stroke="#9a3412" stroke-width="1" />
        
        <circle cx="170" cy="220" r="21" fill="url(#ceramicGrad)" stroke="#9a3412" stroke-width="1" />
        <circle cx="280" cy="230" r="17" fill="url(#ceramicGrad)" stroke="#9a3412" stroke-width="1" />
        <circle cx="390" cy="215" r="23" fill="url(#ceramicGrad)" stroke="#9a3412" stroke-width="1" />

        <!-- LITHIUM CONDUCTION PATHWAYS -->
        <path d="M 130 130 C 145 170, 155 180, 170 220 C 185 240, 210 260, 230 277" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="4,2" />
        <path d="M 360 130 C 350 170, 370 185, 390 215 C 410 240, 420 260, 430 277" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="4,2" />

        <!-- LOWER DOUBLE LAYER -->
        <rect x="70" y="277" width="460" height="12" fill="#fef9c3" stroke="#eab308" stroke-dasharray="3,3" stroke-width="1" />
        <g fill="#b45309" font-family="monospace" font-size="10" font-weight="bold">
            <text x="110" y="287">Li⁺</text><text x="180" y="287">Li⁺</text><text x="250" y="287">Li⁺</text>
            <text x="320" y="287">Li⁺</text><text x="390" y="287">Li⁺</text><text x="460" y="287">Li⁺</text>
        </g>

        <!-- BOTTOM CURRENT COLLECTOR -->
        <rect x="70" y="289" width="460" height="45" rx="4" fill="url(#steelGrad)" stroke="#475569" stroke-width="1.5" />
        <text x="210" y="317" font-family="-apple-system,BlinkMacSystemFont" font-size="12" font-weight="700" fill="#0f172a">
            LOWER STAINLESS STEEL ELECTRODE (BLOCKING)
        </text>
        <line x1="300" y1="334" x2="300" y2="365" stroke="#0f172a" stroke-width="3" />
        <circle cx="300" cy="367" r="4" fill="#0284c7" />

        <!-- LEGEND & SPECS -->
        <g transform="translate(560, 70)">
            <rect x="0" y="0" width="260" height="264" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2" />
            <text x="15" y="25" font-family="-apple-system,BlinkMacSystemFont" font-size="12" font-weight="700" fill="#0f172a">MICROSCOPIC PHENOMENA</text>
            
            <circle cx="20" cy="50" r="5" fill="#10b981" />
            <text x="32" y="54" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">Electrolyte Matrix (Bulk)</text>
            <text x="32" y="68" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Ohmic conduction pathway (R1)</text>

            <circle cx="20" cy="95" r="5" fill="#f97316" />
            <text x="32" y="99" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">Ceramic Fillers / Grains</text>
            <text x="32" y="113" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Inorganic ion-conducting particles</text>

            <line x1="15" y1="135" x2="25" y2="135" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="3,1" />
            <text x="32" y="139" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">Interphase / Grain Boundary</text>
            <text x="32" y="153" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Space-charge relaxation (R2 || C1)</text>

            <rect x="15" y="172" width="10" height="10" fill="#fef9c3" stroke="#eab308" />
            <text x="32" y="181" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">Blocking Metal Interface</text>
            <text x="32" y="195" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Double-layer capacitive tail (CPE1)</text>

            <line x1="15" y1="212" x2="245" y2="212" stroke="#e2e8f0" stroke-width="1" />
            <text x="15" y="232" font-family="monospace" font-size="11" font-weight="700" fill="#0284c7">{r_b_text}</text>
            <text x="15" y="250" font-family="monospace" font-size="11" font-weight="700" fill="#d97706">{r_hf_text}</text>
        </g>
    </svg>
    """

def render_equivalent_circuit_svg(r1=None, r2=None, c1=None, q1=None, a1=None):
    r1_val = f"{r1:.3f} Ω" if r1 else "R1 (R_bulk)"
    r2_val = f"{r2:.3f} Ω" if r2 else "R2 (R_gb)"
    c1_val = f"{c1*1e9:.2f} nF" if c1 else "C1"
    cpe_val = f"Q={q1:.2e}, α={a1:.3f}" if q1 else "CPE1 (Blocking Tail)"
    
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 260" width="100%" height="220" style="background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;">
        <text x="30" y="32" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="13" font-weight="700" fill="#0f172a">
            RELAXIS-STANDARD EQUIVALENT ELECTRICAL CIRCUIT: R1 + (R2 || C1) + CPE1
        </text>

        <circle cx="50" cy="130" r="5" fill="#0f172a" />
        <line x1="55" y1="130" x2="90" y2="130" stroke="#0f172a" stroke-width="2.5" />
        <text x="35" y="115" font-family="monospace" font-size="11" font-weight="600" fill="#475569">Port 1</text>

        <!-- R1 -->
        <rect x="90" y="118" width="70" height="24" rx="2" fill="#e0f2fe" stroke="#0284c7" stroke-width="2" />
        <text x="125" y="134" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#0369a1" text-anchor="middle">R1</text>
        <text x="125" y="160" font-family="monospace" font-size="10" font-weight="600" fill="#0369a1" text-anchor="middle">{r1_val}</text>

        <!-- Parallel Branch Rails -->
        <line x1="160" y1="130" x2="220" y2="130" stroke="#0f172a" stroke-width="2.5" />
        <circle cx="220" cy="130" r="3.5" fill="#0f172a" />
        <line x1="220" y1="130" x2="220" y2="80" stroke="#0f172a" stroke-width="2" />
        <line x1="220" y1="80" x2="270" y2="80" stroke="#0f172a" stroke-width="2" />
        <line x1="220" y1="130" x2="220" y2="180" stroke="#0f172a" stroke-width="2" />
        <line x1="220" y1="180" x2="270" y2="180" stroke="#0f172a" stroke-width="2" />

        <!-- C1 -->
        <line x1="270" y1="80" x2="295" y2="80" stroke="#0f172a" stroke-width="2" />
        <line x1="295" y1="65" x2="295" y2="95" stroke="#059669" stroke-width="3" />
        <line x1="305" y1="65" x2="305" y2="95" stroke="#059669" stroke-width="3" />
        <line x1="305" y1="80" x2="330" y2="80" stroke="#0f172a" stroke-width="2" />
        <text x="300" y="58" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#059669" text-anchor="middle">C1</text>
        <text x="300" y="110" font-family="monospace" font-size="10" font-weight="600" fill="#059669" text-anchor="middle">{c1_val}</text>

        <!-- R2 -->
        <rect x="270" y="168" width="60" height="24" rx="2" fill="#fef3c7" stroke="#d97706" stroke-width="2" />
        <text x="300" y="184" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#b45309" text-anchor="middle">R2</text>
        <text x="300" y="210" font-family="monospace" font-size="10" font-weight="600" fill="#b45309" text-anchor="middle">{r2_val}</text>

        <!-- Recombine -->
        <line x1="330" y1="80" x2="380" y2="80" stroke="#0f172a" stroke-width="2" />
        <line x1="330" y1="180" x2="380" y2="180" stroke="#0f172a" stroke-width="2" />
        <line x1="380" y1="80" x2="380" y2="130" stroke="#0f172a" stroke-width="2" />
        <line x1="380" y1="180" x2="380" y2="130" stroke="#0f172a" stroke-width="2" />
        <circle cx="380" cy="130" r="3.5" fill="#0f172a" />

        <line x1="380" y1="130" x2="440" y2="130" stroke="#0f172a" stroke-width="2.5" />

        <!-- CPE1 -->
        <line x1="440" y1="130" x2="465" y2="130" stroke="#0f172a" stroke-width="2" />
        <line x1="465" y1="115" x2="465" y2="145" stroke="#7c3aed" stroke-width="3" />
        <path d="M 477 115 Q 471 130 477 145" fill="none" stroke="#7c3aed" stroke-width="3" />
        <line x1="475" y1="130" x2="520" y2="130" stroke="#0f172a" stroke-width="2" />
        <text x="472" y="105" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#7c3aed" text-anchor="middle">CPE1 (Blocking Tail)</text>
        <text x="472" y="165" font-family="monospace" font-size="10" font-weight="600" fill="#7c3aed" text-anchor="middle">{cpe_val}</text>

        <line x1="520" y1="130" x2="560" y2="130" stroke="#0f172a" stroke-width="2.5" />
        <circle cx="560" cy="130" r="5" fill="#0f172a" />
        <text x="548" y="115" font-family="monospace" font-size="11" font-weight="600" fill="#475569">Port 2</text>
    </svg>
    """


# ==============================================================================
# 6. UNIVERSAL FILE INGESTION ENGINE
# ==============================================================================

def parse_any_eis_file(file_name, file_bytes):
    """
    Universal sniffer & parser:
    - BioLogic (.mpt) with multi-line metadata headers
    - Gamry (.dta)
    - Autolab / Metrohm (.csv, .txt)
    - Excel single-table (.xlsx, .xls)
    - Excel multi-run horizontal tiled sheets (like EIS ALL.xlsx)
    """
    name_lower = file_name.lower()
    
    # Text, BioLogic MPT, or CSV
    if name_lower.endswith(('.mpt', '.txt', '.csv', '.dta')):
        try:
            content = file_bytes.decode('utf-8', errors='ignore')
        except:
            content = str(file_bytes)
            
        lines = content.splitlines()
        header_idx = 0
        for i, line in enumerate(lines[:120]):
            ll = line.lower()
            if ('freq' in ll or 'f/hz' in ll) and ("z'" in ll or "re(z)" in ll or "real" in ll):
                header_idx = i
                break
                
        df = pd.read_csv(
            io.StringIO("\n".join(lines[header_idx:])),
            sep=None, engine='python'
        )
        
        f_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['freq', 'f/hz', 'frequency'])), None)
        re_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["z'", "re(z)", "real", "z_re"])), None)
        im_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["-z''", "z''", "im(z)", "-im(z)", "imag"])), None)
        
        if f_col and re_col and im_col:
            f = pd.to_numeric(df[f_col], errors='coerce').values
            re = pd.to_numeric(df[re_col], errors='coerce').values
            im = np.abs(pd.to_numeric(df[im_col], errors='coerce').values)
            
            valid = (f > 0) & (~np.isnan(f)) & (~np.isnan(re)) & (~np.isnan(im))
            f, re, im = f[valid], re[valid], im[valid]
            sort_idx = np.argsort(f)[::-1]
            clean_name = file_name.rsplit('.', 1)[0]
            return {clean_name: {"f": f[sort_idx], "z_re": re[sort_idx], "z_im": im[sort_idx]}}
            
    # Excel Workbooks
    elif name_lower.endswith(('.xlsx', '.xls')):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        runs = {}
        
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            cols_lower = [str(c).lower() for c in df.columns]
            freq_col_count = sum(1 for c in cols_lower if 'freq' in c)
            
            # Case A: Horizontal tiled multi-run sheet
            if freq_col_count > 1:
                col_i = 0
                while col_i < len(df.columns):
                    hdr = str(df.columns[col_i]).strip()
                    sub_cols = df.columns[col_i:min(col_i+6, len(df.columns))]
                    f_c = next((c for c in sub_cols if 'freq' in str(c).lower()), None)
                    r_c = next((c for c in sub_cols if "z'" in str(c).lower() or "re" in str(c).lower()), None)
                    i_c = next((c for c in sub_cols if "z''" in str(c).lower() or "im" in str(c).lower()), None)
                    
                    if f_c and r_c and i_c:
                        sub = df[[f_c, r_c, i_c]].dropna()
                        f = pd.to_numeric(sub[f_c], errors='coerce').values
                        r = pd.to_numeric(sub[r_c], errors='coerce').values
                        m = np.abs(pd.to_numeric(sub[i_c], errors='coerce').values)
                        valid = (f > 0) & (~np.isnan(f)) & (~np.isnan(r)) & (~np.isnan(m))
                        lbl = hdr if not hdr.startswith("Unnamed") else f"{sheet}_Run_{col_i//7 + 1}"
                        sort_idx = np.argsort(f[valid])[::-1]
                        runs[lbl] = {"f": f[valid][sort_idx], "z_re": r[valid][sort_idx], "z_im": m[valid][sort_idx]}
                        col_i += 7
                    else:
                        col_i += 1
                        
            # Case B: Standard Single-Table Sheet
            elif any('freq' in c for c in cols_lower) and any("z'" in c or "re" in c for c in cols_lower):
                f_c = next(c for c in df.columns if 'freq' in str(c).lower())
                r_c = next(c for c in df.columns if "z'" in str(c).lower() or "re" in str(c).lower())
                i_c = next(c for c in df.columns if "z''" in str(c).lower() or "im" in str(c).lower())
                sub = df[[f_c, r_c, i_c]].dropna()
                f = pd.to_numeric(sub[f_c], errors='coerce').values
                r = pd.to_numeric(sub[r_c], errors='coerce').values
                m = np.abs(pd.to_numeric(sub[i_c], errors='coerce').values)
                valid = (f > 0) & (~np.isnan(f)) & (~np.isnan(r)) & (~np.isnan(m))
                sort_idx = np.argsort(f[valid])[::-1]
                runs[sheet] = {"f": f[valid][sort_idx], "z_re": r[valid][sort_idx], "z_im": m[valid][sort_idx]}
                
        return runs
    return {}

def extract_temperature_from_string(text):
    """Sniffs temperature from filename (e.g., 'sample_25C.mpt', 'run_298K.csv')."""
    match_c = re.search(r'([+-]?\d+\.?\d*)\s*(?:°?C|degc)', text, re.IGNORECASE)
    if match_c:
        return float(match_c.group(1))
    match_k = re.search(r'([+-]?\d+\.?\d*)\s*K', text, re.IGNORECASE)
    if match_k:
        return float(match_k.group(1)) - 273.15
    return None


# ==============================================================================
# 7. MAIN STREAMLIT APPLICATION INTERFACE
# ==============================================================================

st.sidebar.title("⚡ Universal EIS Suite")
st.sidebar.markdown("**Universal Solid-State Blocking Cell Diagnostic Engine**")

with st.sidebar.expander("📐 Cell Geometry & Setup", expanded=True):
    thickness_um = st.number_input("Membrane Thickness (µm)", min_value=0.1, max_value=10000.0, value=155.0, step=5.0)
    diameter_mm = st.number_input("Electrode Diameter (mm)", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
    
    thickness_cm = thickness_um * 1e-4
    area_cm2 = np.pi * ((diameter_mm / 2.0) * 0.1) ** 2
    cell_constant = thickness_cm / area_cm2
    st.info(f"Cell Factor ($L/A$): **{cell_constant:.4f} cm⁻¹**\nArea: **{area_cm2:.4f} cm²**")

with st.sidebar.expander("📁 Ingest EIS Data", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload EIS Sweeps (.mpt, .csv, .txt, .xlsx)",
        type=["mpt", "csv", "txt", "xlsx", "xls", "dta"],
        accept_multiple_files=True,
        help="Upload single or multiple files from BioLogic, Gamry, Autolab, or Excel."
    )
    benchmark_file = st.file_uploader("Optional: RelaxIS Export Benchmark (.xlsx)", type=["xlsx", "xls"])

# Ingest all files into master runs dictionary
master_runs = {}
if uploaded_files:
    for uploaded_f in uploaded_files:
        f_bytes = uploaded_f.read()
        f_runs = parse_any_eis_file(uploaded_f.name, f_bytes)
        master_runs.update(f_runs)

if master_runs:
    st.sidebar.success(f"Loaded **{len(master_runs)}** measurement series.")
    
    # Parse RelaxIS benchmark if present
    relaxis_benchmarks = {}
    if benchmark_file:
        try:
            b_bytes = benchmark_file.read()
            xls_b = pd.ExcelFile(io.BytesIO(b_bytes))
            for s in xls_b.sheet_names:
                df_b = pd.read_excel(xls_b, sheet_name=s)
                # Check for parameter columns
                for c in range(len(df_b.columns) - 1):
                    if "resistance 1" in str(df_b.columns[c]).lower():
                        r1 = float(df_b.columns[c+1])
                        r2 = float(df_b.iloc[0, c+1])
                        relaxis_benchmarks[s] = {"R1": r1, "R2": r2, "R_total": r1 + r2}
        except:
            pass

    # TABS INTERFACE
    tab_single, tab_bode, tab_drt, tab_batch, tab_arrhenius, tab_schematics = st.tabs([
        "🔬 Single-Run Deconvolution",
        "📊 Dual-Axis Bode Diagnostics",
        "⚡ DRT Relaxation Spectrum",
        "📑 Batch Processing & Validation",
        "🔥 Arrhenius Ionic Transport",
        "📐 Cell & Circuit Architecture"
    ])

    # TAB 1: SINGLE-RUN DECONVOLUTION
    with tab_single:
        col_sel, col_btn = st.columns([2, 1])
        with col_sel:
            selected_run = st.selectbox("Select Measurement Series", list(master_runs.keys()))
            cur_data = master_runs[selected_run]
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            auto_fit_clicked = st.button("🚀 Run Dynamic EEC Fit", type="primary", use_container_width=True)
            
        col_metrics, col_nyquist = st.columns([1, 2])
        
        fit_res = None
        if auto_fit_clicked or f"fit_{selected_run}" in st.session_state:
            with st.spinner("Executing non-linear least squares deconvolution..."):
                fit_res = fit_universal_blocking_spectrum(cur_data['f'], cur_data['z_re'], cur_data['z_im'])
                st.session_state[f"fit_{selected_run}"] = fit_res
        elif f"fit_{selected_run}" in st.session_state:
            fit_res = st.session_state[f"fit_{selected_run}"]

        with col_metrics:
            if fit_res and fit_res["success"]:
                sigma_bulk = cell_constant / fit_res["R1_bulk"]
                sigma_total = cell_constant / fit_res["R_total"]
                
                st.markdown("#### Physical Transport Parameters")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">Bulk Ohmic Resistance (R1)</div>
                    <div class="metric-value">{fit_res['R1_bulk']:.3f} <span class="metric-unit">Ω</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">σ_bulk: <b>{sigma_bulk:.3e} S/cm</b></div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">HF Interphase / GB Resistance (R2)</div>
                    <div class="metric-value">{fit_res['R2_hf']:.3f} <span class="metric-unit">Ω</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">C1: <b>{fit_res['C1']*1e9:.2f} nF</b> | τ: <b>{fit_res['tau_s']*1e6:.3f} µs</b></div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">Total Resistance (R_tot)</div>
                    <div class="metric-value">{fit_res['R_total']:.3f} <span class="metric-unit">Ω</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Total Ionic Conductivity: <b>{sigma_total:.3e} S/cm</b></div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">Blocking CPE Exponent (α_dl)</div>
                    <div class="metric-value">{fit_res['alpha_dl']:.3f} <span class="metric-unit">(dimensionless)</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Q_dl: <b>{fit_res['Q_dl']:.2e} S·s^α</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Click **'Run Dynamic EEC Fit'** to optimize parameters.")
                st.markdown(f"""
                **Series Metadata:**
                - Spectrum Points: **{len(cur_data['f'])}**
                - Frequency Range: **{np.min(cur_data['f']):.2e} Hz to {np.max(cur_data['f']):.2e} Hz**
                - HF Real Intercept: **{np.min(cur_data['z_re']):.3f} Ω**
                - Maximum LF |Z|: **{np.max(cur_data['z_im']):.1f} Ω**
                """)

        with col_nyquist:
            fig_nyq = go.Figure()
            fig_nyq.add_trace(go.Scatter(
                x=cur_data['z_re'], y=cur_data['z_im'],
                mode='markers', name='Experimental',
                marker=dict(size=7, color='#0f172a', symbol='circle-open', line=dict(width=1.5, color='#0284c7')),
                hovertext=[f"f = {frq:.2e} Hz<br>Z' = {re:.2f} Ω<br>-Z'' = {im:.2f} Ω" for frq, re, im in zip(cur_data['f'], cur_data['z_re'], cur_data['z_im'])],
                hoverinfo="text"
            ))
            
            if fit_res and fit_res["success"]:
                fig_nyq.add_trace(go.Scatter(
                    x=fit_res['Z_re_fit'], y=fit_res['Z_im_fit'],
                    mode='lines', name='Universal Model: R1 + (R2||C1) + CPE1',
                    line=dict(color='#d97706', width=2.5)
                ))
                
            fig_nyq.update_layout(
                title=f"Nyquist Plot: {selected_run}",
                xaxis_title="Real Impedance Z' (Ω)",
                yaxis_title="Negative Imaginary Impedance -Z'' (Ω)",
                template="plotly_white", height=520,
                yaxis=dict(scaleanchor="x", scaleratio=1),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_nyq, use_container_width=True)

    # TAB 2: BODE & KRAMERS-KRONIG DIAGNOSTICS
    with tab_bode:
        st.subheader("Dual-Axis Bode Magnitude & Phase Spectroscopy")
        cur_data = master_runs[selected_run]
        z_mag = np.sqrt(cur_data['z_re']**2 + cur_data['z_im']**2)
        phase_deg = np.arctan2(cur_data['z_im'], cur_data['z_re']) * (180.0 / np.pi)
        
        fig_bode = make_subplots(specs=[[{"secondary_y": True}]])
        fig_bode.add_trace(go.Scatter(x=cur_data['f'], y=z_mag, mode='markers', name='|Z| Exp', marker=dict(color='#0284c7', size=6)), secondary_y=False)
        fig_bode.add_trace(go.Scatter(x=cur_data['f'], y=phase_deg, mode='markers', name='Phase -θ Exp', marker=dict(color='#dc2626', size=6)), secondary_y=True)
        
        if f"fit_{selected_run}" in st.session_state and st.session_state[f"fit_{selected_run}"]["success"]:
            fo = st.session_state[f"fit_{selected_run}"]
            fig_bode.add_trace(go.Scatter(x=fo['f_dense'], y=fo['Z_mag_fit'], mode='lines', name='|Z| Fit', line=dict(color='#0284c7', width=1.5)), secondary_y=False)
            fig_bode.add_trace(go.Scatter(x=fo['f_dense'], y=fo['phase_fit'], mode='lines', name='Phase Fit', line=dict(color='#dc2626', width=1.5, dash='dash')), secondary_y=True)
            
        fig_bode.update_xaxes(title_text="Frequency (Hz)", type="log")
        fig_bode.update_yaxes(title_text="Impedance Magnitude |Z| (Ω)", type="log", secondary_y=False)
        fig_bode.update_yaxes(title_text="Phase Angle -θ (°)", secondary_y=True)
        fig_bode.update_layout(template="plotly_white", height=460, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(fig_bode, use_container_width=True)
        
        st.markdown("#### Linear Kramers-Kronig (Lin-KK) Validation")
        kk = compute_lin_kk_residuals(cur_data['f'], cur_data['z_re'], cur_data['z_im'])
        fig_kk = go.Figure()
        fig_kk.add_trace(go.Scatter(x=cur_data['f'], y=kk['delta_re_pct'], mode='lines+markers', name='Δ Re(Z) / |Z| (%)', line=dict(color='#0284c7')))
        fig_kk.add_trace(go.Scatter(x=cur_data['f'], y=kk['delta_im_pct'], mode='lines+markers', name='Δ Im(Z) / |Z| (%)', line=dict(color='#d97706')))
        fig_kk.update_xaxes(title_text="Frequency (Hz)", type="log")
        fig_kk.update_yaxes(title_text="K-K Residual (%)")
        fig_kk.update_layout(title=f"K-K Residual Spectrum (Max Residual: {kk['max_residual_pct']:.2f}%)", template="plotly_white", height=320)
        st.plotly_chart(fig_kk, use_container_width=True)

    # TAB 3: DRT DECONVOLUTION
    with tab_drt:
        st.subheader("Model-Free Distribution of Relaxation Times (DRT)")
        lambda_val = st.slider("Regularization Parameter (log10 λ)", min_value=-5.0, max_value=1.0, value=-2.0, step=0.2)
        drt_out = compute_drt(cur_data['f'], cur_data['z_re'], cur_data['z_im'], lambda_reg=10**lambda_val)
        
        fig_drt = go.Figure()
        fig_drt.add_trace(go.Scatter(x=drt_out['tau'], y=drt_out['gamma'], mode='lines', fill='tozeroy', line=dict(color='#7c3aed', width=2), fillcolor='rgba(124, 58, 237, 0.15)', name='DRT γ(ln τ)'))
        fig_drt.update_xaxes(title_text="Relaxation Time τ (s)", type="log")
        fig_drt.update_yaxes(title_text="Distribution Function γ(ln τ) [Ω]")
        fig_drt.update_layout(title="Continuous DRT Spectrum", template="plotly_white", height=460)
        st.plotly_chart(fig_drt, use_container_width=True)

    # TAB 4: BATCH PROCESSING
    with tab_batch:
        st.subheader("Batch Deconvolution Engine")
        if st.button("⚡ Process All Loaded Runs", type="primary"):
            b_list = []
            pbar = st.progress(0)
            for i, (name, d) in enumerate(master_runs.items()):
                res = fit_universal_blocking_spectrum(d['f'], d['z_re'], d['z_im'])
                if res["success"]:
                    sigma_t = cell_constant / res["R_total"]
                    b_list.append({
                        "Run Name": name,
                        "R1_Bulk (Ω)": round(res["R1_bulk"], 3),
                        "R2_HF (Ω)": round(res["R2_hf"], 3),
                        "R_total (Ω)": round(res["R_total"], 3),
                        "σ_total (S/cm)": f"{sigma_t:.3e}",
                        "C1 (nF)": round(res["C1"] * 1e9, 2),
                        "f_relax (kHz)": round(res["f_relax_kHz"], 1),
                        "CPE_alpha": round(res["alpha_dl"], 3)
                    })
                pbar.progress((i + 1) / len(master_runs))
            b_df = pd.DataFrame(b_list)
            st.session_state["universal_batch_df"] = b_df
            
        if "universal_batch_df" in st.session_state:
            st.dataframe(st.session_state["universal_batch_df"], use_container_width=True)
            csv_buf = io.StringIO()
            st.session_state["universal_batch_df"].to_csv(csv_buf, index=False)
            st.download_button("📥 Download Results CSV", csv_buf.getvalue(), "universal_eis_results.csv", "text/csv")

    # TAB 5: ARRHENIUS IONIC TRANSPORT
    with tab_arrhenius:
        st.subheader("Arrhenius Transport Extraction")
        arr_cols = st.columns(3)
        temp_dict = {}
        with st.expander("Assign Temperatures (°C) to Runs", expanded=True):
            for i, name in enumerate(master_runs.keys()):
                col = arr_cols[i % 3]
                sniffed_t = extract_temperature_from_string(name)
                default_t = sniffed_t if sniffed_t is not None else 25.0 + (i * 2.5)
                temp_dict[name] = col.number_input(f"{name} (°C)", value=float(default_t), key=f"t_in_{name}")
                
        if st.button("📈 Compute Arrhenius Activation Energy", type="primary"):
            pts = []
            for name, d in master_runs.items():
                res = fit_universal_blocking_spectrum(d['f'], d['z_re'], d['z_im'])
                if res["success"]:
                    T_k = temp_dict[name] + 273.15
                    s_t = cell_constant / res["R_total"]
                    pts.append({"Run": name, "T_C": temp_dict[name], "inv_T": 1000.0 / T_k, "ln_sT": np.log(s_t * T_k)})
                    
            arr_df = pd.DataFrame(pts)
            slope, intercept, r_val, _, _ = linregress(arr_df['inv_T'], arr_df['ln_sT'])
            k_B = 8.617333262145e-5
            E_a_eV = -slope * 1000.0 * k_B
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Activation Energy Ea", f"{E_a_eV:.3f} eV")
            m2.metric("Activation Energy", f"{(E_a_eV * 96.485):.2f} kJ/mol")
            m3.metric("Correlation R²", f"{r_val**2:.4f}")
            
            fig_arr = go.Figure()
            fig_arr.add_trace(go.Scatter(x=arr_df['inv_T'], y=arr_df['ln_sT'], mode='markers', text=[f"{r['Run']}: {r['T_C']}°C" for _, r in arr_df.iterrows()], name='Data'))
            x_line = np.linspace(arr_df['inv_T'].min(), arr_df['inv_T'].max(), 50)
            fig_arr.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode='lines', line=dict(color='#d97706', dash='dash'), name='Linear Fit'))
            fig_arr.update_layout(title="Arrhenius Transport: ln(σT) vs 1000/T", xaxis_title="1000 / T (K⁻¹)", yaxis_title="ln(σ · T) [S·K/cm]", template="plotly_white", height=480)
            st.plotly_chart(fig_arr, use_container_width=True)

    # TAB 6: SCHEMATICS
    with tab_schematics:
        st.subheader("Architectural & Equivalent Circuit Visualizations")
        cur_fit = st.session_state.get(f"fit_{selected_run}", None)
        r_b_val = cur_fit["R1_bulk"] if cur_fit and cur_fit["success"] else None
        r_hf_val = cur_fit["R2_hf"] if cur_fit and cur_fit["success"] else None
        c1_val = cur_fit["C1"] if cur_fit and cur_fit["success"] else None
        q1_val = cur_fit["Q_dl"] if cur_fit and cur_fit["success"] else None
        a1_val = cur_fit["alpha_dl"] if cur_fit and cur_fit["success"] else None
        
        st.markdown("#### 1. Cross-Sectional Physical Blocking Cell")
        st.markdown(render_cell_schematic_svg(thickness_um, diameter_mm, r_b_val, r_hf_val), unsafe_allow_html=True)
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 2. Equivalent Electrical Circuit Topology")
        st.markdown(render_equivalent_circuit_svg(r_b_val, r_hf_val, c1_val, q1_val, a1_val), unsafe_allow_html=True)

else:
    st.info("👈 Upload your EIS files in the sidebar (BioLogic `.mpt`, Gamry `.dta`, generic `.csv`, or Excel workbooks) to begin.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Physical Cell Model")
        st.markdown(render_cell_schematic_svg(155.0, 10.0), unsafe_allow_html=True)
    with col2:
        st.markdown("#### Equivalent Circuit Topology")
        st.markdown(render_equivalent_circuit_svg(), unsafe_allow_html=True)
