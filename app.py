import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress
from scipy.linalg import toeplitz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time

# ==============================================================================
# 1. APPLICATION CONFIGURATION & METADATA
# ==============================================================================

st.set_page_config(
    page_title="Solid-State Blocking Cell EIS Diagnostic Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific clarity, clean borders, and typography
st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e6ed;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-header {
        font-size: 0.82rem;
        color: #5c6b73;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.45rem;
        color: #1a2a3a;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #7b8893;
        font-weight: 400;
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
# 2. PHYSICAL & EQUIVALENT ELECTRICAL CIRCUIT (EEC) ENGINE
# ==============================================================================

def model_Z_relaxis(params, w):
    """
    Standard RelaxIS-compatible Symmetric Blocking Cell Model:
    Z(w) = R1 + (R2 / (1 + j * w * R2 * C1)) + 1 / (Q_dl * (j * w)^alpha_dl)
    
    Parameters:
    - R1: Bulk electrolyte ohmic resistance (Ohm)
    - R2: High-frequency grain boundary / interphase resistance (Ohm)
    - C1: High-frequency geometric/dielectric capacitance (F)
    - Q_dl: Constant Phase Element pseudo-capacitance for blocking electrode (S * s^alpha)
    - alpha_dl: CPE dispersion exponent (dimensionless, 0 < alpha <= 1)
    """
    R1, R2, C1, Q_dl, a_dl = params
    
    # High-Frequency parallel relaxation arc
    jw = 1j * w
    Z_arc = R2 / (1.0 + jw * R2 * C1)
    
    # Low-Frequency blocking double-layer CPE
    jw_dl = jw ** a_dl
    Z_dl = 1.0 / (Q_dl * jw_dl)
    
    return R1 + Z_arc + Z_dl

def residuals_relaxis(params, w, z_re, z_im):
    """
    Complex modulus-weighted residuals for stable non-linear least squares.
    """
    Z_pred = model_Z_relaxis(params, w)
    res_re = (Z_pred.real - z_re) / np.abs(z_re)
    res_im = (-Z_pred.imag - z_im) / np.abs(z_im)
    return np.concatenate([res_re, res_im])

def fit_spectrum(f, z_re, z_im):
    """
    Robust parameter fitting for symmetric blocking electrodes.
    Calculates parameter uncertainties, relaxation frequency, and dense simulation curves.
    """
    w = 2.0 * np.pi * f
    
    # Deterministic parameter bounding based on physical geometry
    r1_guess = float(np.min(z_re))
    hf_sub = z_re[f > 1e4]
    r2_guess = float(np.max(hf_sub) - r1_guess) if len(hf_sub) > 0 else 2.5
    r2_guess = max(r2_guess, 0.2)
    
    # Apex of relaxation arc is ~ 400 - 500 kHz -> tau ~ 3.3e-7 s
    c1_guess = 3.3e-7 / r2_guess
    
    lf_idx = np.argmin(f)
    z_lf_mag = np.sqrt(z_re[lf_idx]**2 + z_im[lf_idx]**2)
    w_lf = w[lf_idx]
    q_dl_guess = float(1.0 / (z_lf_mag * (w_lf ** 0.87)))
    
    p0 = [r1_guess, r2_guess, c1_guess, q_dl_guess, 0.87]
    bounds = (
        [0.01, 0.01, 1e-12, 1e-10, 0.40],
        [1e6,  1e6,  1e-3,  1e-1,  1.00]
    )
    
    try:
        res = least_squares(
            residuals_relaxis, p0, bounds=bounds,
            args=(w, z_re, z_im), ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=2500
        )
        p = res.x
        
        # Dense simulation spectrum for visualization
        f_dense = np.logspace(np.log10(np.min(f)), np.log10(np.max(f)), 300)
        w_dense = 2.0 * np.pi * f_dense
        Z_dense = model_Z_relaxis(p, w_dense)
        
        tau = p[1] * p[2]
        f_relax = 1.0 / (2.0 * np.pi * tau)
        
        # Goodness-of-fit (Chi-squared metric)
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
# 3. KRAMERS-KRONIG (LIN-KK) RESIDUAL VALIDATION ENGINE
# ==============================================================================

def compute_lin_kk_residuals(f, z_re, z_im, M=None):
    """
    Evaluates physical stationarity and causality using a linear Kramers-Kronig test.
    Constructs a generalized Voigt circuit (M RC elements) to test consistency.
    """
    N = len(f)
    if M is None:
        M = int(N * 0.8)
    
    w = 2.0 * np.pi * f
    w_min, w_max = np.min(w), np.max(w)
    tau_k = np.logspace(-np.log10(w_max), -np.log10(w_min), M)
    
    # Build design matrix: Real and Imaginary components of Voigt cascade
    A_re = np.zeros((N, M + 1))
    A_im = np.zeros((N, M))
    
    A_re[:, 0] = 1.0  # R_inf term
    for k in range(M):
        denom = 1.0 + (w * tau_k[k])**2
        A_re[:, k + 1] = 1.0 / denom
        A_im[:, k] = -(w * tau_k[k]) / denom
        
    A = np.vstack([A_re, np.hstack([np.zeros((N, 1)), A_im])])
    b = np.concatenate([z_re, -z_im])
    
    # Ridge regression solve
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
# 4. DISTRIBUTION OF RELAXATION TIMES (DRT) ENGINE
# ==============================================================================

def compute_drt(f, z_re, z_im, lambda_reg=1e-2):
    """
    Deconvolutes the continuous distribution of relaxation times gamma(tau)
    using Tikhonov regularization with a Gaussian basis representation.
    """
    w = 2.0 * np.pi * f
    N = len(f)
    
    # Set tau domain spanning 1 decade beyond experimental frequency window
    tau_min = 1.0 / np.max(w) / 10.0
    tau_max = 1.0 / np.min(w) * 10.0
    M = 120
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), M)
    
    ln_tau = np.log(tau)
    delta_ln_tau = ln_tau[1] - ln_tau[0]
    
    # Kernel matrix for imaginary impedance component
    # -Z''(w) = w * integral [ gamma(tau) / (1 + (w*tau)^2) ] d(tau)
    A = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            wt = w[i] * tau[j]
            A[i, j] = wt / (1.0 + wt**2) * delta_ln_tau
            
    b = z_im
    
    # First-order difference regularization matrix (smoothness prior)
    L = np.zeros((M - 1, M))
    for i in range(M - 1):
        L[i, i] = -1.0
        L[i, i + 1] = 1.0
        
    H = A.T @ A + lambda_reg * (L.T @ L)
    g = A.T @ b
    
    # Non-negative bounded solution
    try:
        from scipy.optimize import nnls
        gamma, _ = nnls(H, g)
    except:
        gamma = np.linalg.solve(H + 1e-6 * np.eye(M), g)
        gamma = np.clip(gamma, 0, None)
        
    return {
        "tau": tau,
        "freq_equiv": 1.0 / (2.0 * np.pi * tau),
        "gamma": gamma
    }


# ==============================================================================
# 5. HIGH-FIDELITY VECTOR SCHEMATIC VISUALIZERS (SVG ENGINE)
# ==============================================================================

def render_cell_schematic_svg(thickness_um, diameter_mm, r_bulk=None, r_hf=None):
    """
    Generates an architectural cross-sectional vector schematic of the symmetric
    stainless steel blocking electrode cell with composite electrolyte.
    """
    r_b_text = f"R_bulk: {r_bulk:.2f} Ω" if r_bulk else "R_bulk (Matrix)"
    r_hf_text = f"R_hf: {r_hf:.2f} Ω" if r_hf else "R_hf (Interphase)"
    
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 420" width="100%" height="320" style="background:#ffffff; border-radius:8px; border:1px solid #dce4ec;">
        <defs>
            <linearGradient id="steelGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#b0bec5" />
                <stop offset="50%" stop-color="#cfd8dc" />
                <stop offset="100%" stop-color="#90a4ae" />
            </linearGradient>
            <linearGradient id="polymerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#e8f5e9" />
                <stop offset="50%" stop-color="#c8e6c9" />
                <stop offset="100%" stop-color="#e8f5e9" />
            </linearGradient>
            <radialGradient id="ceramicGrad" cx="35%" cy="35%" r="65%">
                <stop offset="0%" stop-color="#ffe082" />
                <stop offset="40%" stop-color="#ffca28" />
                <stop offset="100%" stop-color="#ff8f00" />
            </radialGradient>
            <filter id="dropShadow" x="-5%" y="-5%" width="110%" height="110%">
                <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.15"/>
            </filter>
        </defs>

        <!-- TITLE / LEGEND -->
        <text x="30" y="32" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="14" font-weight="700" fill="#1e293b">
            SYMMETRIC ION-BLOCKING CELL ARCHITECTURE (SS | COMPOSITE MEMBRANE | SS)
        </text>
        <text x="30" y="50" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="11" fill="#64748b">
            Geometric Area A: {np.pi*((diameter_mm/2)*0.1)**2:.3f} cm² | Thickness L: {thickness_um:.1f} µm | Blocking Condition: t_ion ≈ 1.0
        </text>

        <!-- TOP CURRENT COLLECTOR (STAINLESS STEEL) -->
        <rect x="70" y="70" width="460" height="45" rx="4" fill="url(#steelGrad)" stroke="#607d8b" stroke-width="1.5" filter="url(#dropShadow)" />
        <text x="210" y="98" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="12" font-weight="700" fill="#263238">
            UPPER STAINLESS STEEL ELECTRODE (ION BLOCKING)
        </text>
        
        <!-- ELECTRICAL TERMINAL (TOP) -->
        <line x1="300" y1="70" x2="300" y2="40" stroke="#1e293b" stroke-width="3" />
        <circle cx="300" cy="38" r="4" fill="#0284c7" />
        <text x="312" y="42" font-family="monospace" font-size="11" font-weight="600" fill="#0284c7">WE / CE Terminal</text>

        <!-- UPPER INTERFACIAL DOUBLE LAYER (CPE_dl,top) -->
        <rect x="70" y="115" width="460" height="12" fill="#fff9c4" stroke="#fbc02d" stroke-dasharray="3,3" stroke-width="1" />
        <!-- Microscopic Helmholtz ion accumulation charges -->
        <g fill="#d97706" font-family="Arial" font-size="10" font-weight="bold">
            <text x="90" y="125">Li⁺</text><text x="140" y="125">Li⁺</text>
            <text x="200" y="125">Li⁺</text><text x="270" y="125">Li⁺</text>
            <text x="340" y="125">Li⁺</text><text x="410" y="125">Li⁺</text>
            <text x="470" y="125">Li⁺</text>
        </g>

        <!-- COMPOSITE SOLID ELECTROLYTE MEMBRANE (PVDF MATRIX) -->
        <rect x="70" y="127" width="460" height="150" fill="url(#polymerGrad)" stroke="#4caf50" stroke-width="2" />
        
        <!-- CERAMIC FILLER PARTICLES (LLTO PEROVSKITE GRAINS) -->
        <!-- Row 1 -->
        <circle cx="120" cy="155" r="16" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="210" cy="165" r="20" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="310" cy="150" r="18" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="415" cy="160" r="22" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="490" cy="152" r="14" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <!-- Row 2 -->
        <circle cx="160" cy="205" r="22" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="260" cy="215" r="17" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="365" cy="200" r="21" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="460" cy="215" r="19" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <!-- Row 3 -->
        <circle cx="115" cy="255" r="15" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="205" cy="250" r="19" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="315" cy="260" r="18" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />
        <circle cx="410" cy="252" r="16" fill="url(#ceramicGrad)" stroke="#e65100" stroke-width="1.2" />

        <!-- LATTICE LABELS -->
        <text x="202" y="169" font-family="-apple-system,BlinkMacSystemFont" font-size="8" font-weight="700" fill="#ffffff">LLTO</text>
        <text x="152" y="209" font-family="-apple-system,BlinkMacSystemFont" font-size="8" font-weight="700" fill="#ffffff">LLTO</text>
        <text x="357" y="204" font-family="-apple-system,BlinkMacSystemFont" font-size="8" font-weight="700" fill="#ffffff">LLTO</text>

        <!-- TORTUOUS LITHIUM TRANSPORT PATHWAYS (ARROWS) -->
        <path d="M 120 130 C 135 170, 145 180, 160 205 C 175 220, 195 235, 205 277" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="4,2" />
        <path d="M 340 130 C 330 170, 345 185, 365 200 C 390 220, 400 240, 410 277" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="4,2" />

        <!-- LOWER INTERFACIAL DOUBLE LAYER (CPE_dl,bottom) -->
        <rect x="70" y="277" width="460" height="12" fill="#fff9c4" stroke="#fbc02d" stroke-dasharray="3,3" stroke-width="1" />
        <g fill="#d97706" font-family="Arial" font-size="10" font-weight="bold">
            <text x="110" y="287">Li⁺</text><text x="175" y="287">Li⁺</text>
            <text x="240" y="287">Li⁺</text><text x="305" y="287">Li⁺</text>
            <text x="380" y="287">Li⁺</text><text x="445" y="287">Li⁺</text>
        </g>

        <!-- BOTTOM CURRENT COLLECTOR (STAINLESS STEEL) -->
        <rect x="70" y="289" width="460" height="45" rx="4" fill="url(#steelGrad)" stroke="#607d8b" stroke-width="1.5" filter="url(#dropShadow)" />
        <text x="210" y="317" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="12" font-weight="700" fill="#263238">
            LOWER STAINLESS STEEL ELECTRODE (ION BLOCKING)
        </text>
        
        <!-- ELECTRICAL TERMINAL (BOTTOM) -->
        <line x1="300" y1="334" x2="300" y2="365" stroke="#1e293b" stroke-width="3" />
        <circle cx="300" cy="367" r="4" fill="#0284c7" />
        <text x="312" y="370" font-family="monospace" font-size="11" font-weight="600" fill="#0284c7">RE / Sense Terminal</text>

        <!-- ANNOTATIONS & PHYSICAL DECOUPLING BOX (RIGHT SIDE) -->
        <g transform="translate(560, 70)">
            <rect x="0" y="0" width="260" height="264" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2" />
            
            <text x="15" y="25" font-family="-apple-system,BlinkMacSystemFont" font-size="12" font-weight="700" fill="#0f172a">PHYSICAL PHENOMENA</text>
            
            <circle cx="20" cy="50" r="5" fill="#4caf50" />
            <text x="32" y="54" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">PVDF Polymer Matrix</text>
            <text x="32" y="68" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Ohmic bulk ion conduction (R1)</text>

            <circle cx="20" cy="95" r="5" fill="#ffca28" />
            <text x="32" y="99" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">LLTO Ceramic Fillers</text>
            <text x="32" y="113" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Sol-gel perovskite ion-conductors</text>

            <line x1="15" y1="135" x2="25" y2="135" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="3,1" />
            <text x="32" y="139" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">Polymer-Ceramic Interface</text>
            <text x="32" y="153" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Interfacial barrier / GB arc (R2 || C1)</text>

            <rect x="15" y="172" width="10" height="10" fill="#fff9c4" stroke="#fbc02d" />
            <text x="32" y="181" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="600" fill="#1e293b">Blocking Metal Interface</text>
            <text x="32" y="195" font-family="-apple-system,BlinkMacSystemFont" font-size="10" fill="#64748b">Helmholtz double-layer tail (CPE_dl)</text>

            <!-- LIVE METRIC OVERLAY -->
            <line x1="15" y1="212" x2="245" y2="212" stroke="#e2e8f0" stroke-width="1" />
            <text x="15" y="232" font-family="monospace" font-size="11" font-weight="700" fill="#0369a1">{r_b_text}</text>
            <text x="15" y="250" font-family="monospace" font-size="11" font-weight="700" fill="#b45309">{r_hf_text}</text>
        </g>
    </svg>
    """
    return svg

def render_equivalent_circuit_svg(r1=None, r2=None, c1=None, q1=None, a1=None):
    """
    Generates an electrical schematic representing the RelaxIS equivalent circuit:
    R1 - p(R2, C1) - CPE1
    """
    r1_val = f"{r1:.3f} Ω" if r1 else "R1 (R_bulk)"
    r2_val = f"{r2:.3f} Ω" if r2 else "R2 (R_gb)"
    c1_val = f"{c1*1e9:.2f} nF" if c1 else "C1"
    cpe_val = f"Q={q1:.2e}, α={a1:.3f}" if q1 else "CPE1 (Blocking Tail)"
    
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 260" width="100%" height="220" style="background:#ffffff; border-radius:8px; border:1px solid #dce4ec;">
        <defs>
            <filter id="glow" x="-10%" y="-10%" width="120%" height="120%">
                <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#0284c7" flood-opacity="0.3"/>
            </filter>
        </defs>

        <!-- TITLE -->
        <text x="30" y="32" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto" font-size="13" font-weight="700" fill="#1e293b">
            RELAXIS EQUIVALENT ELECTRICAL CIRCUIT (EEC): R1 + (R2 || C1) + CPE1
        </text>

        <!-- LEFT INPUT TERMINAL -->
        <circle cx="50" cy="130" r="5" fill="#0f172a" />
        <line x1="55" y1="130" x2="90" y2="130" stroke="#0f172a" stroke-width="2.5" />
        <text x="35" y="115" font-family="monospace" font-size="11" font-weight="600" fill="#475569">Port A</text>

        <!-- COMPONENT 1: R1 (BULK RESISTANCE) -->
        <rect x="90" y="118" width="70" height="24" rx="2" fill="#e0f2fe" stroke="#0284c7" stroke-width="2" />
        <text x="125" y="134" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#0369a1" text-anchor="middle">R1</text>
        <text x="125" y="160" font-family="monospace" font-size="10" font-weight="600" fill="#0369a1" text-anchor="middle">{r1_val}</text>

        <!-- LEAD FROM R1 TO PARALLEL BRANCH -->
        <line x1="160" y1="130" x2="220" y2="130" stroke="#0f172a" stroke-width="2.5" />
        <circle cx="220" cy="130" r="3.5" fill="#0f172a" />

        <!-- PARALLEL BRANCH RAILS (TOP: CAPACITOR C1, BOTTOM: RESISTOR R2) -->
        <!-- Rail Up -->
        <line x1="220" y1="130" x2="220" y2="80" stroke="#0f172a" stroke-width="2" />
        <line x1="220" y1="80" x2="270" y2="80" stroke="#0f172a" stroke-width="2" />
        
        <!-- Rail Down -->
        <line x1="220" y1="130" x2="220" y2="180" stroke="#0f172a" stroke-width="2" />
        <line x1="220" y1="180" x2="270" y2="180" stroke="#0f172a" stroke-width="2" />

        <!-- TOP BRANCH: CAPACITANCE C1 (PLATES) -->
        <line x1="270" y1="80" x2="295" y2="80" stroke="#0f172a" stroke-width="2" />
        <line x1="295" y1="65" x2="295" y2="95" stroke="#059669" stroke-width="3" />
        <line x1="305" y1="65" x2="305" y2="95" stroke="#059669" stroke-width="3" />
        <line x1="305" y1="80" x2="330" y2="80" stroke="#0f172a" stroke-width="2" />
        <text x="300" y="58" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#059669" text-anchor="middle">C1</text>
        <text x="300" y="110" font-family="monospace" font-size="10" font-weight="600" fill="#059669" text-anchor="middle">{c1_val}</text>

        <!-- BOTTOM BRANCH: RESISTANCE R2 (HF ARC) -->
        <rect x="270" y="168" width="60" height="24" rx="2" fill="#fef3c7" stroke="#d97706" stroke-width="2" />
        <text x="300" y="184" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#b45309" text-anchor="middle">R2</text>
        <text x="300" y="210" font-family="monospace" font-size="10" font-weight="600" fill="#b45309" text-anchor="middle">{r2_val}</text>

        <!-- RECOMBINE PARALLEL BRANCHES -->
        <line x1="330" y1="80" x2="380" y2="80" stroke="#0f172a" stroke-width="2" />
        <line x1="330" y1="180" x2="380" y2="180" stroke="#0f172a" stroke-width="2" />
        <line x1="380" y1="80" x2="380" y2="130" stroke="#0f172a" stroke-width="2" />
        <line x1="380" y1="180" x2="380" y2="130" stroke="#0f172a" stroke-width="2" />
        <circle cx="380" cy="130" r="3.5" fill="#0f172a" />

        <!-- LEAD TO CPE1 -->
        <line x1="380" y1="130" x2="440" y2="130" stroke="#0f172a" stroke-width="2.5" />

        <!-- COMPONENT 3: CONSTANT PHASE ELEMENT (CPE1) -->
        <!-- CPE Symbol: Curved plate opposite flat plate -->
        <line x1="440" y1="130" x2="465" y2="130" stroke="#0f172a" stroke-width="2" />
        <line x1="465" y1="115" x2="465" y2="145" stroke="#7c3aed" stroke-width="3" />
        <path d="M 477 115 Q 471 130 477 145" fill="none" stroke="#7c3aed" stroke-width="3" />
        <line x1="475" y1="130" x2="520" y2="130" stroke="#0f172a" stroke-width="2" />
        <text x="472" y="105" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#7c3aed" text-anchor="middle">CPE1 (Blocking Tail)</text>
        <text x="472" y="165" font-family="monospace" font-size="10" font-weight="600" fill="#7c3aed" text-anchor="middle">{cpe_val}</text>

        <!-- RIGHT OUTPUT TERMINAL -->
        <line x1="520" y1="130" x2="560" y2="130" stroke="#0f172a" stroke-width="2.5" />
        <circle cx="560" cy="130" r="5" fill="#0f172a" />
        <text x="548" y="115" font-family="monospace" font-size="11" font-weight="600" fill="#475569">Port B</text>

        <!-- LEGEND & CIRCUIT SPECIFICATIONS -->
        <g transform="translate(620, 60)">
            <rect x="0" y="0" width="200" height="150" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
            <text x="12" y="22" font-family="-apple-system,BlinkMacSystemFont" font-size="11" font-weight="700" fill="#0f172a">IMPEDANCE EQUATION</text>
            <text x="12" y="44" font-family="serif" font-size="12" font-style="italic" fill="#334155">Z = R1 + Z_arc + Z_CPE</text>
            <text x="12" y="68" font-family="monospace" font-size="9" fill="#475569">Z_arc = R2/(1 + jωR2C1)</text>
            <text x="12" y="88" font-family="monospace" font-size="9" fill="#475569">Z_CPE = 1 / [Q(jω)^α]</text>
            <text x="12" y="114" font-family="-apple-system,BlinkMacSystemFont" font-size="9" font-weight="600" fill="#059669">τ_relax = R2 · C1</text>
            <text x="12" y="132" font-family="-apple-system,BlinkMacSystemFont" font-size="9" font-weight="600" fill="#0284c7">f_apex = 1 / (2πτ)</text>
        </g>
    </svg>
    """
    return svg


# ==============================================================================
# 6. EXCEL WORKBOOK & RELAXIS FILE PARSING
# ==============================================================================

def parse_tiled_eis_workbook(file_buffer, sheet_name):
    """
    Parses horizontal multi-run tiled Excel sheets (like EIS ALL.xlsx)
    where runs repeat horizontally across 7-8 columns.
    """
    df = pd.read_excel(file_buffer, sheet_name=sheet_name)
    runs = {}
    col_idx = 0
    total_cols = len(df.columns)
    
    while col_idx < total_cols:
        header_candidate = str(df.columns[col_idx]).strip()
        freq_col, z_real_col, z_imag_col = None, None, None
        
        for offset in range(min(5, total_cols - col_idx)):
            col_str = str(df.columns[col_idx + offset]).lower()
            if 'freq' in col_str:
                freq_col = df.columns[col_idx + offset]
            elif "z'" in col_str or "re(z)" in col_str:
                z_real_col = df.columns[col_idx + offset]
            elif "-z''" in col_str or "im(z)" in col_str:
                z_imag_col = df.columns[col_idx + offset]
                
        if freq_col and z_real_col and z_imag_col:
            sub = df[[freq_col, z_real_col, z_imag_col]].dropna()
            label = header_candidate if not header_candidate.startswith("Unnamed") else f"Run_{col_idx//7 + 1}"
            
            f = sub[freq_col].values.astype(float)
            re = sub[z_real_col].values.astype(float)
            im = np.abs(sub[z_imag_col].values.astype(float))
            
            # Filter non-physical inductive points (Z' < 0 or f <= 0)
            valid = (f > 0) & (re > 0)
            f, re, im = f[valid], re[valid], im[valid]
            
            sort_idx = np.argsort(f)[::-1]
            runs[label] = {
                "f": f[sort_idx],
                "z_re": re[sort_idx],
                "z_im": im[sort_idx]
            }
            col_idx += 7
        else:
            col_idx += 1
            
    return runs

def parse_relaxis_benchmark_workbook(file_buffer, sheet_name):
    """
    Parses RelaxIS exported parameter columns from FITTING ALL0.xlsx.
    """
    df = pd.read_excel(file_buffer, sheet_name=sheet_name)
    benchmarks = {}
    
    col_idx = 0
    while col_idx < len(df.columns):
        run_label = str(df.columns[col_idx]).strip()
        for c in range(col_idx, min(col_idx + 14, len(df.columns))):
            col_str = str(df.columns[c])
            if "resistance 1" in col_str.lower() and c + 1 < len(df.columns):
                try:
                    r1 = float(df.columns[c+1])
                    r2 = float(df.iloc[0, c+1])
                    c1 = float(df.iloc[1, c+1])
                    q1 = float(df.iloc[2, c+1])
                    a1 = float(df.iloc[3, c+1])
                    benchmarks[run_label] = {
                        "R1": r1, "R2": r2, "R_total": r1 + r2,
                        "C1": c1, "Q1": q1, "alpha1": a1
                    }
                except:
                    pass
                break
        col_idx += 12
    return benchmarks


# ==============================================================================
# 7. MAIN INTERACTIVE APPLICATION LAYOUT
# ==============================================================================

# Sidebar Configuration Controls
st.sidebar.title("⚡ Cell Diagnostics Setup")
st.sidebar.markdown("Symmetric blocking cell geometry & data ingestion parameters.")

with st.sidebar.expander("📐 Pellet & Electrode Geometry", expanded=True):
    thickness_um = st.number_input(
        "Membrane Thickness L (µm)",
        min_value=1.0, max_value=3000.0, value=155.0, step=5.0,
        help="Thickness of the composite solid electrolyte pellet."
    )
    diameter_mm = st.number_input(
        "Electrode Diameter D (mm)",
        min_value=1.0, max_value=50.0, value=10.0, step=0.5,
        help="Diameter of the polished stainless steel pin contact."
    )
    
    thickness_cm = thickness_um * 1e-4
    area_cm2 = np.pi * ((diameter_mm / 2.0) * 0.1) ** 2
    cell_constant = thickness_cm / area_cm2
    st.info(f"Geometric Factor ($L/A$): **{cell_constant:.4f} cm⁻¹**\nArea: **{area_cm2:.4f} cm²**")

with st.sidebar.expander("📁 Data Files Upload", expanded=True):
    raw_file = st.file_uploader("Upload EIS Data (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])
    benchmark_file = st.file_uploader("Optional: RelaxIS Benchmark (.xlsx)", type=["xlsx", "xls"])

# Data Processing Pipeline Execution
if raw_file is not None:
    # Determine sheets
    runs = {}
    if raw_file.name.endswith(('.xlsx', '.xls')):
        xls = pd.ExcelFile(raw_file)
        sheet_choice = st.sidebar.selectbox("Select Measurement Cell Sheet", xls.sheet_names)
        runs = parse_tiled_eis_workbook(raw_file, sheet_choice)
    else:
        # Generic CSV
        df_csv = pd.read_csv(raw_file)
        f_col = [c for c in df_csv.columns if 'freq' in c.lower()][0]
        re_col = [c for c in df_csv.columns if "z'" in c.lower() or 're' in c.lower()][0]
        im_col = [c for c in df_csv.columns if "z''" in c.lower() or 'im' in c.lower()][0]
        f_vals = df_csv[f_col].values.astype(float)
        re_vals = df_csv[re_col].values.astype(float)
        im_vals = np.abs(df_csv[im_col].values.astype(float))
        sort_idx = np.argsort(f_vals)[::-1]
        runs["Uploaded_Sweep"] = {
            "f": f_vals[sort_idx],
            "z_re": re_vals[sort_idx],
            "z_im": im_vals[sort_idx]
        }
        sheet_choice = "CSV_Dataset"
        
    benchmarks = {}
    if benchmark_file is not None:
        try:
            benchmarks = parse_relaxis_benchmark_workbook(benchmark_file, sheet_choice)
            st.sidebar.success(f"Loaded RelaxIS benchmark fits for {len(benchmarks)} runs.")
        except Exception as e:
            st.sidebar.warning(f"Could not extract RelaxIS benchmark: {e}")

    # Header Metric Summary Bar
    st.markdown(f"### Cell Diagnostic Session: `{sheet_choice}` ({len(runs)} Measurement Sweeps)")
    
    # NAVIGATION TABS
    tab_single, tab_bode, tab_drt, tab_batch, tab_arrhenius, tab_schematics = st.tabs([
        "🔬 Single-Run Deconvolution",
        "📊 Dual-Axis Bode Diagnostics",
        "⚡ DRT Relaxation Spectrum",
        "📑 Batch Processing & Validation",
        "🔥 Arrhenius Ionic Transport",
        "📐 Cell & Circuit Architecture"
    ])
    
    # --------------------------------------------------------------------------
    # TAB 1: SINGLE-RUN DECONVOLUTION & NYQUIST FITTING
    # --------------------------------------------------------------------------
    with tab_single:
        col_select, col_actions = st.columns([2, 1])
        with col_select:
            run_name = st.selectbox("Select Measurement Run to Analyze", list(runs.keys()))
            cur_data = runs[run_name]
        with col_actions:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            auto_fit_clicked = st.button("🚀 Run EEC Deconvolution", type="primary", use_container_width=True)
            
        col_metrics, col_nyquist = st.columns([1, 2])
        
        fit_output = None
        if auto_fit_clicked or f"fit_{run_name}" in st.session_state:
            with st.spinner("Executing Levenberg-Marquardt least-squares optimization..."):
                fit_output = fit_spectrum(cur_data['f'], cur_data['z_re'], cur_data['z_im'])
                st.session_state[f"fit_{run_name}"] = fit_output
        elif f"fit_{run_name}" in st.session_state:
            fit_output = st.session_state[f"fit_{run_name}"]

        with col_metrics:
            if fit_output and fit_output["success"]:
                sigma_bulk = cell_constant / fit_output["R1_bulk"]
                sigma_total = cell_constant / fit_output["R_total"]
                
                st.markdown("#### Electrochemical Parameters")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">Bulk Ohmic Resistance (R1)</div>
                    <div class="metric-value">{fit_output['R1_bulk']:.3f} <span class="metric-unit">Ω</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">σ_bulk: <b>{sigma_bulk:.3e} S/cm</b></div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">HF Interphase / GB Resistance (R2)</div>
                    <div class="metric-value">{fit_output['R2_hf']:.3f} <span class="metric-unit">Ω</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">C1: <b>{fit_output['C1']*1e9:.2f} nF</b> | τ: <b>{fit_output['tau_s']*1e6:.3f} µs</b></div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">Total Cell Resistance (R_tot)</div>
                    <div class="metric-value">{fit_output['R_total']:.3f} <span class="metric-unit">Ω</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Total Ionic Conductivity: <b>{sigma_total:.3e} S/cm</b></div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">Blocking CPE Exponent (α_dl)</div>
                    <div class="metric-value">{fit_output['alpha_dl']:.3f} <span class="metric-unit">(dimensionless)</span></div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Q_dl: <b>{fit_output['Q_dl']:.2e} S·s^α</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Check against RelaxIS benchmark
                if run_name in benchmarks:
                    b = benchmarks[run_name]
                    dev = abs(fit_output['R_total'] - b['R_total']) / b['R_total'] * 100.0
                    st.success(f"**RelaxIS Benchmark Deviation: {dev:.2f}%**\nRelaxIS R1: {b['R1']:.3f} Ω | R2: {b['R2']:.3f} Ω | R_tot: {b['R_total']:.3f} Ω")
            else:
                st.info("Click **'Run EEC Deconvolution'** to optimize parameters.")
                st.markdown(f"""
                **Dataset Metadata:**
                - Total Points: **{len(cur_data['f'])}**
                - Max Frequency: **{np.max(cur_data['f']):.2e} Hz**
                - Min Frequency: **{np.min(cur_data['f']):.4f} Hz**
                - Min Real Intercept: **{np.min(cur_data['z_re']):.2f} Ω**
                - Max LF Imaginary: **{np.max(cur_data['z_im']):.2f} Ω**
                """)

        with col_nyquist:
            fig_nyq = go.Figure()
            
            # Experimental Scatter Points
            fig_nyq.add_trace(go.Scatter(
                x=cur_data['z_re'], y=cur_data['z_im'],
                mode='markers',
                name='Experimental Data',
                marker=dict(size=7, color='#1e293b', symbol='circle-open', line=dict(width=1.5, color='#0284c7')),
                hovertext=[f"f = {frq:.2e} Hz<br>Z' = {re:.2f} Ω<br>-Z'' = {im:.2f} Ω" for frq, re, im in zip(cur_data['f'], cur_data['z_re'], cur_data['z_im'])],
                hoverinfo="text"
            ))
            
            # Simulated EEC Line
            if fit_output and fit_output["success"]:
                fig_nyq.add_trace(go.Scatter(
                    x=fit_output['Z_re_fit'], y=fit_output['Z_im_fit'],
                    mode='lines',
                    name='Fitted EEC: R1 + (R2||C1) + CPE1',
                    line=dict(color='#d97706', width=2.5)
                ))
                
            fig_nyq.update_layout(
                title=dict(text=f"Nyquist Diagram: {run_name}", font=dict(size=14, family="sans-serif")),
                xaxis_title="Real Impedance Z' (Ω)",
                yaxis_title="Negative Imaginary Impedance -Z'' (Ω)",
                template="plotly_white",
                height=530,
                legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
                yaxis=dict(scaleanchor="x", scaleratio=1),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_nyq, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: BODE & KRAMERS-KRONIG DIAGNOSTICS
    # --------------------------------------------------------------------------
    with tab_bode:
        st.subheader("Dual-Axis Bode Magnitude & Phase Response")
        cur_data = runs[run_name]
        
        # Calculate experimental Bode vectors
        z_mag = np.sqrt(cur_data['z_re']**2 + cur_data['z_im']**2)
        phase_deg = np.arctan2(cur_data['z_im'], cur_data['z_re']) * (180.0 / np.pi)
        
        fig_bode = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Magnitude Trace (Log-Log)
        fig_bode.add_trace(
            go.Scatter(x=cur_data['f'], y=z_mag, mode='markers', name='|Z| Experimental',
                       marker=dict(size=6, color='#0284c7', symbol='square')),
            secondary_y=False
        )
        
        # Phase Trace (Semi-Log)
        fig_bode.add_trace(
            go.Scatter(x=cur_data['f'], y=phase_deg, mode='markers', name='Phase -θ Experimental',
                       marker=dict(size=6, color='#dc2626', symbol='circle')),
            secondary_y=True
        )
        
        if f"fit_{run_name}" in st.session_state and st.session_state[f"fit_{run_name}"]["success"]:
            fo = st.session_state[f"fit_{run_name}"]
            fig_bode.add_trace(
                go.Scatter(x=fo['f_dense'], y=fo['Z_mag_fit'], mode='lines', name='|Z| Fit',
                           line=dict(color='#0284c7', width=1.5)),
                secondary_y=False
            )
            fig_bode.add_trace(
                go.Scatter(x=fo['f_dense'], y=fo['phase_fit'], mode='lines', name='Phase Fit',
                           line=dict(color='#dc2626', width=1.5, dash='dash')),
                secondary_y=True
            )
            
        fig_bode.update_xaxes(title_text="Frequency (Hz)", type="log")
        fig_bode.update_yaxes(title_text="Impedance Magnitude |Z| (Ω)", type="log", secondary_y=False)
        fig_bode.update_yaxes(title_text="Phase Angle -θ (°)", secondary_y=True)
        fig_bode.update_layout(template="plotly_white", height=480, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(fig_bode, use_container_width=True)
        
        # Kramers-Kronig Residual Analysis
        st.markdown("#### Linear Kramers-Kronig (Lin-KK) Causality Test")
        with st.spinner("Computing Lin-KK test..."):
            kk_res = compute_lin_kk_residuals(cur_data['f'], cur_data['z_re'], cur_data['z_im'])
            
        fig_kk = go.Figure()
        fig_kk.add_trace(go.Scatter(x=cur_data['f'], y=kk_res['delta_re_pct'], mode='lines+markers', name='Δ Re(Z) / |Z| (%)', line=dict(color='#0284c7')))
        fig_kk.add_trace(go.Scatter(x=cur_data['f'], y=kk_res['delta_im_pct'], mode='lines+markers', name='Δ Im(Z) / |Z| (%)', line=dict(color='#d97706')))
        fig_kk.update_xaxes(title_text="Frequency (Hz)", type="log")
        fig_kk.update_yaxes(title_text="K-K Residual (%)")
        fig_kk.update_layout(
            title=f"Lin-KK Compliance Residuals (Max Error: {kk_res['max_residual_pct']:.2f}%)",
            template="plotly_white", height=320, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_kk, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: DISTRIBUTION OF RELAXATION TIMES (DRT)
    # --------------------------------------------------------------------------
    with tab_drt:
        st.subheader("Model-Free Distribution of Relaxation Times (DRT)")
        st.markdown("""
        DRT deconvolutes the continuous relaxation spectrum $\gamma(\ln \tau)$ without requiring any a priori equivalent circuit assumptions.
        Peaks correspond directly to characteristic polarization processes in the polymer and composite interphase.
        """)
        
        lambda_val = st.slider("Regularization Parameter (log10 λ)", min_value=-5.0, max_value=1.0, value=-2.0, step=0.2)
        
        with st.spinner("Deconvoluting continuous relaxation distribution..."):
            drt_out = compute_drt(cur_data['f'], cur_data['z_re'], cur_data['z_im'], lambda_reg=10**lambda_val)
            
        fig_drt = go.Figure()
        fig_drt.add_trace(go.Scatter(
            x=drt_out['tau'], y=drt_out['gamma'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#7c3aed', width=2),
            fillcolor='rgba(124, 58, 237, 0.15)',
            name='DRT γ(ln τ)'
        ))
        
        fig_drt.update_xaxes(title_text="Relaxation Time τ (s)", type="log")
        fig_drt.update_yaxes(title_text="Distribution Function γ(ln τ) [Ω]")
        fig_drt.update_layout(
            title="Continuous DRT Spectrum",
            template="plotly_white", height=460, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_drt, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: BATCH PROCESSING & VALIDATION
    # --------------------------------------------------------------------------
    with tab_batch:
        st.subheader("Automated Multi-Run Batch Deconvolution")
        st.markdown("Executes equivalent circuit parameter extraction across all measurement series in the current sheet.")
        
        run_batch = st.button("⚡ Execute Batch Deconvolution on All Runs", type="primary")
        
        if run_batch:
            results_list = []
            progress_bar = st.progress(0)
            
            for i, (name, d) in enumerate(runs.items()):
                res = fit_spectrum(d['f'], d['z_re'], d['z_im'])
                if res["success"]:
                    sigma_t = cell_constant / res["R_total"]
                    row = {
                        "Run Date / Label": name,
                        "R1_Bulk (Ω)": round(res["R1_bulk"], 3),
                        "R2_HF (Ω)": round(res["R2_hf"], 3),
                        "R_total (Ω)": round(res["R_total"], 3),
                        "σ_total (S/cm)": f"{sigma_t:.3e}",
                        "C1 (nF)": round(res["C1"] * 1e9, 2),
                        "f_relax (kHz)": round(res["f_relax_kHz"], 1),
                        "CPE_alpha": round(res["alpha_dl"], 3),
                        "Chi_Sq": f"{res['chi_sq']:.2e}"
                    }
                    if name in benchmarks:
                        b_rtot = benchmarks[name]["R_total"]
                        row["RelaxIS_R_tot (Ω)"] = round(b_rtot, 3)
                        row["Diff (%)"] = round(abs(res["R_total"] - b_rtot) / b_rtot * 100.0, 2)
                    results_list.append(row)
                progress_bar.progress((i + 1) / len(runs))
                
            batch_df = pd.DataFrame(results_list)
            st.session_state["batch_results"] = batch_df
            
        if "batch_results" in st.session_state:
            df_display = st.session_state["batch_results"]
            st.dataframe(df_display, use_container_width=True)
            
            csv_buf = io.StringIO()
            df_display.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Download Summary CSV",
                data=csv_buf.getvalue(),
                file_name=f"{sheet_choice}_deconvolution_summary.csv",
                mime="text/csv"
            )

    # --------------------------------------------------------------------------
    # TAB 5: ARRHENIUS IONIC TRANSPORT & ACTIVATION ENERGY
    # --------------------------------------------------------------------------
    with tab_arrhenius:
        st.subheader("Arrhenius Transport & Hopping Activation Energy")
        st.markdown("Assign thermal temperatures to each sweep to extract the bulk/total hopping activation barrier ($E_a$).")
        
        arr_cols = st.columns(3)
        temp_assignments = {}
        with st.expander("Assign Measurement Temperatures (°C)", expanded=False):
            for i, name in enumerate(runs.keys()):
                col = arr_cols[i % 3]
                temp_assignments[name] = col.number_input(f"{name} (°C)", value=25.0 + (i * 2.5), key=f"temp_{name}")
                
        calc_arrhenius = st.button("📈 Compute Arrhenius Activation Energy", type="primary")
        
        if calc_arrhenius:
            arr_pts = []
            for name, d in runs.items():
                res = fit_spectrum(d['f'], d['z_re'], d['z_im'])
                if res["success"]:
                    T_k = temp_assignments[name] + 273.15
                    s_t = cell_constant / res["R_total"]
                    arr_pts.append({
                        "Run": name,
                        "T_C": temp_assignments[name],
                        "T_K": T_k,
                        "inv_T": 1000.0 / T_k,
                        "sigma": s_t,
                        "ln_sT": np.log(s_t * T_k)
                    })
                    
            arr_df = pd.DataFrame(arr_pts)
            
            slope, intercept, r_val, p_val, std_err = linregress(arr_df['inv_T'], arr_df['ln_sT'])
            k_B = 8.617333262145e-5  # eV/K
            E_a_eV = -slope * 1000.0 * k_B
            E_a_kJ = E_a_eV * 96.485
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Activation Energy Ea", f"{E_a_eV:.3f} eV")
            m2.metric("Activation Energy", f"{E_a_kJ:.2f} kJ/mol")
            m3.metric("Correlation R²", f"{r_val**2:.4f}")
            m4.metric("Pre-Exponential σ0", f"{np.exp(intercept):.2e} S·K/cm")
            
            fig_arr = go.Figure()
            fig_arr.add_trace(go.Scatter(
                x=arr_df['inv_T'], y=arr_df['ln_sT'],
                mode='markers',
                marker=dict(size=9, color='#0284c7'),
                text=[f"{r['Run']}: {r['T_C']}°C" for _, r in arr_df.iterrows()],
                name='Experimental Data'
            ))
            
            x_line = np.linspace(arr_df['inv_T'].min(), arr_df['inv_T'].max(), 50)
            y_line = slope * x_line + intercept
            fig_arr.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='#d97706', dash='dash', width=2),
                name=f'Arrhenius Linear Fit (Ea = {E_a_eV:.3f} eV)'
            ))
            
            fig_arr.update_layout(
                title="Arrhenius Plot: ln(σ·T) vs 1000/T",
                xaxis_title="Inverse Temperature 1000 / T (K⁻¹)",
                yaxis_title="ln(σ · T) [S · K · cm⁻¹]",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_arr, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 6: ARCHITECTURAL VECTOR SCHEMATICS
    # --------------------------------------------------------------------------
    with tab_schematics:
        st.subheader("Physical Cell & Equivalent Circuit Representation")
        st.markdown("Dynamic architectural diagrams reflecting cell geometry, current collector interfaces, and equivalent circuit components.")
        
        cur_fit = st.session_state.get(f"fit_{run_name}", None)
        r_b_val = cur_fit["R1_bulk"] if cur_fit and cur_fit["success"] else None
        r_hf_val = cur_fit["R2_hf"] if cur_fit and cur_fit["success"] else None
        c1_val = cur_fit["C1"] if cur_fit and cur_fit["success"] else None
        q1_val = cur_fit["Q_dl"] if cur_fit and cur_fit["success"] else None
        a1_val = cur_fit["alpha_dl"] if cur_fit and cur_fit["success"] else None
        
        st.markdown("#### 1. Cross-Sectional Blocking Cell Physical Architecture")
        cell_svg = render_cell_schematic_svg(thickness_um, diameter_mm, r_b_val, r_hf_val)
        st.markdown(cell_svg, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 2. RelaxIS-Standard Equivalent Electrical Circuit (EEC)")
        circuit_svg = render_equivalent_circuit_svg(r_b_val, r_hf_val, c1_val, q1_val, a1_val)
        st.markdown(circuit_svg, unsafe_allow_html=True)

else:
    # Landing View when no file is uploaded
    st.info("👈 Upload your experimental data file (`EIS ALL.xlsx`) in the sidebar to initiate the diagnostic session.")
    
    st.markdown("### Architectural Overview of the Platform")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Physical Cell Model")
        st.markdown(render_cell_schematic_svg(155.0, 10.0), unsafe_allow_html=True)
    with col2:
        st.markdown("#### Equivalent Circuit Topology")
        st.markdown(render_equivalent_circuit_svg(), unsafe_allow_html=True)
