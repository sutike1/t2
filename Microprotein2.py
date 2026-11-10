"""
SIMULATOR INTEGRAT DE MICROPROTEINE – model didactic de fermentatie batch
Titlu proiect: Simularea procesului de producție a micoproteinelor prin fermentație cu Fusarium venenatum
Autor: Jánosi-Rancz Júlia (Biotehnologii industriale I.)
"""

import csv
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp

# Streamlit oldal beállításai
st.set_page_config(page_title="Simularea Producției de Micoproteină", layout="wide")
st.markdown("""
    <style>
       h1 {
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
        }
      
        @media (max-width: 768px) {
            h1 {
                font-size: 1.1rem !important;
                line-height: 1.3 !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

plt.rcParams['text.usetex'] = False  

# ----------------------- PARAMETRI DIDACTICI -----------------------
DEFAULTS = {
    "mu_max": 0.20,       # 1/h, demonstrativ
    "T_opt": 30.0,        # °C, demonstrativ
    "T_sigma": 6.0,       # °C, lățimea curbei termice
    "Ks": 0.0054,         # g/L (F. venenatum glükóz affinitása: 30 µM ≈ 5.4 mg/L)
    "Ko": 0.5,            # mg O2/L
    "Csat": 8.0,          # mg O2/L
    "kLa": 15.0,          # 1/h
    "Yxs": 0.48,          # g biomasă / g substrat
    "qO2": 0.08,          # g O2 / (g biomasă * h)
    "protein_frac": 0.45, # g proteină / g biomasă uscată
    "X0": 0.20,           # g/L
    "S0": 25.0,           # g/L
    "DO0": 8.0,           # mg/L
    "hours": 30.0,        # h
}

SCENARIO_S0 = [5.0, 10.0, 20.0, 30.0, 40.0]


def rhs(t, y, p):
    """y = [X, S, DO]; Monod dublu limitat prin substrat și oxigen."""
    X, S, DO = np.maximum(y, 0.0)

    # Corecție termică gaussiană
    gamma_T = np.exp(-((p["T"] - p["T_opt"]) / p["T_sigma"]) ** 2)

    fS = S / (p["Ks"] + S) if S > 0 else 0.0
    fO = DO / (p["Ko"] + DO) if DO > 0 else 0.0
    mu = p["mu_max"] * gamma_T * fS * fO

    dX = mu * X
    dS = -dX / p["Yxs"]

    # Transfer O2 (mg/L/h) minus consum specific (mg/L/h)
    oxygen_transfer = p["kLa"] * (p["Csat"] - DO)
    oxygen_uptake = p["qO2"] * X * 1000.0
    dDO = oxygen_transfer - oxygen_uptake

    if S <= 0 and dS < 0:
        dS = 0.0
    if DO <= 0 and dDO < 0:
        dDO = 0.0
    return [dX, dS, dDO]


def simulate(p, S0=None, points=200):
    p_sim = p.copy()
    if S0 is not None:
        p_sim["S0"] = float(S0)
    t_eval = np.linspace(0, p_sim["hours"], points)
    y0 = [p_sim["X0"], p_sim["S0"], p_sim["DO0"]]
    
    sol = solve_ivp(
        lambda t, y: rhs(t, y, p_sim),
        (0, p_sim["hours"]), y0, t_eval=t_eval,
        method="Radau", rtol=1e-5, atol=1e-7
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    X, S, DO = np.maximum(sol.y, 0.0)
    protein = X * p_sim["protein_frac"]
    return sol.t, X, S, DO, protein


def summary(p, S0):
    t, X, S, DO, P = simulate(p, S0)
    used = max(S0 - S[-1], 0.0)
    gain = max(X[-1] - p["X0"], 0.0)
    return {
        "S0": S0, "Xf": X[-1], "Sf": S[-1], "DOf": DO[-1],
        "Pf": P[-1], "substrate_used": used,
        "Yxs_observed": gain / used if used > 1e-12 else 0.0
    }

# ----------------------- STREAMLIT UI -----------------------

st.title("Simularea procesului de producție a micoproteinelor prin fermentație cu Fusarium venenatum")
st.markdown("**Autor:** Jánosi-Rancz Júlia | **Specializare:** Biotehnologii industriale I.")

# BAL OLDALI SÁV (SLIDEREK)
st.sidebar.header("Parametrii de Simulare")
if st.sidebar.button("🔄 Resetare Parametri"):
    st.session_state["T"] = DEFAULTS["T_opt"]
    st.session_state["kLa"] = DEFAULTS["kLa"]
    st.session_state["S0"] = DEFAULTS["S0"]
    st.session_state["Ks_mg"] = DEFAULTS["Ks"] * 1000.0
    st.session_state["mu_max"] = DEFAULTS["mu_max"]
    st.session_state["protein_frac"] = DEFAULTS["protein_frac"]
    st.rerun()

p = DEFAULTS.copy()
p["T"] = st.sidebar.slider("Temperatură (°C)", 15.0, 40.0, key="T", value=DEFAULTS["T_opt"])
p["kLa"] = st.sidebar.slider("Aerare kLa (1/h)", 1.0, 200.0, key="kLa", value=DEFAULTS["kLa"])
p["S0"] = st.sidebar.slider("Substrat inițial S_0 (g/L)", 5.0, 50.0, key="S0", value=DEFAULTS["S0"])

ks_mg = st.sidebar.slider(
    "Const. de semisaturație K_s (mg/L)", 
    min_value=0.1, 
    max_value=10.0, 
    value=DEFAULTS["Ks"] * 1000.0, 
    step=0.1,
    key="Ks_mg"
)
p["Ks"] = ks_mg / 1000.0

p["mu_max"] = st.sidebar.slider("μ_max (1/h)", 0.03, 0.35, key="mu_max", value=DEFAULTS["mu_max"])
p["protein_frac"] = st.sidebar.slider("Fracție proteică (g/g)", 0.20, 0.70, key="protein_frac", value=DEFAULTS["protein_frac"])

# SZIMULÁCIÓ LEFUTTATÁSA
t, X, S, DO, P = simulate(p)

# MODEL MATEMATIC EXPLICAT
with st.expander("📌 Vezi Modelul Matematic (Ecuații)", expanded=False):
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.write("**Viteza specifică de creștere:**")
        st.latex(r"\mu = \mu_{\max} \cdot \gamma(T) \cdot \frac{S}{K_s + S} \cdot \frac{DO}{K_o + DO}")
        
        st.write("**Influența temperaturii:**")
        st.latex(r"\gamma(T) = \exp\left( -\left[ \frac{T - T_{\mathrm{opt}}}{T_{\sigma}} \right]^2 \right)")
        
    with col_eq2:
        st.write("**Creșterea biomasei & Consumul substratului:**")
        st.latex(r"\frac{dX}{dt} = \mu \cdot X \quad | \quad \frac{dS}{dt} = -\frac{1}{Y_{X/S}} \cdot \frac{dX}{dt}")
        
        st.write("**Bilanțul oxigenului:**")
        st.latex(r"\frac{dDO}{dt} = k_L a (C_{\mathrm{sat}} - DO) - 1000 \cdot q_{O2} \cdot X")


# EREDMÉNY KIÍRÁSA ZÖLD KÁRTYÁBAN
st.success(
    f"**REZULTAT FINAL (t=30h):** "
    f"Biomasă (X) = **{X[-1]:.2f} g/L** | "
    f"Proteină (P) = **{P[-1]:.2f} g/L** | "
    f"Substrat Rămas (S) = **{S[-1]:.2f} g/L**"
)

# GRAFIKONOK KIRAJZOLÁSA
col1, col2 = st.columns(2)

with col1:
    fig1, ax_growth = plt.subplots(figsize=(5, 2.2))
    ax_growth.plot(t, X, label="Biomasă uscată (X)", color="#2b8a3e", lw=2.2)
    ax_growth.plot(t, P, label="Proteină estimată (P)", color="#099268", lw=2.2, linestyle="--")
    ax_growth.set_title("Creșterea Biomasei și Proteinei", fontsize=9, fontweight="bold")
    ax_growth.set_xlabel("Timp (h)", fontsize=7.5)
    ax_growth.set_ylabel("Concentrație (g/L)", fontsize=7.5)
    ax_growth.tick_params(labelsize=7)
    ax_growth.grid(alpha=0.3)
    ax_growth.legend(fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)  # Memória felszabadítása

    fig2, ax_do = plt.subplots(figsize=(5, 2.2))
    ax_do.plot(t, DO, color="#1971c2", lw=2, label="Oxigen dizolvat (DO)")
    ax_do.axhline(p["Ko"], color="#c92a2a", ls=":", label="Nivel critic (Ko)")
    ax_do.set_title("Bilanțul Oxigenului Dizolvat", fontsize=9, fontweight="bold")
    ax_do.set_xlabel("Timp (h)", fontsize=7.5)
    ax_do.set_ylabel("DO (mg/L)", fontsize=7.5)
    ax_do.tick_params(labelsize=7)
    ax_do.grid(alpha=0.3)
    ax_do.legend(fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)  # Memória felszabadítása

with col2:
    fig3, ax_sub = plt.subplots(figsize=(5, 2.2))
    ax_sub.plot(t, S, color="#e65c00", lw=2.2, label="Substrat (S)")
    ax_sub.set_title("Consumul Substratului", fontsize=9, fontweight="bold")
    ax_sub.set_xlabel("Timp (h)", fontsize=7.5)
    ax_sub.set_ylabel("Substrat (g/L)", fontsize=7.5)
    ax_sub.tick_params(labelsize=7)
    ax_sub.grid(alpha=0.3)
    ax_sub.legend(fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)  # Memória felszabadítása

    fig4, ax_compare = plt.subplots(figsize=(5, 2.2))
    results = [summary(p, s) for s in SCENARIO_S0]
    max_p = max(r["Pf"] for r in results) if results else 1.0
    bars = ax_compare.bar(
        [f"{r['S0']:.0f}" for r in results],
        [r["Pf"] for r in results],
        color="#74c0fc"
    )
    ax_compare.set_title("Proteină finală la diferite S_0 inițiale", fontsize=9, fontweight="bold")
    ax_compare.set_xlabel("Substrat inițial S_0 (g/L)", fontsize=7.5)
    ax_compare.set_ylabel("Proteină (g/L)", fontsize=7.5)
    ax_compare.tick_params(labelsize=7)
    ax_compare.set_ylim(0, max_p * 1.25)
    ax_compare.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, results):
        ax_compare.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + (max_p * 0.02),
            f"{r['Pf']:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold"
        )
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)  # Memória felszabadítása

# EXPORT CSV GOMB STREAMLITBEN
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow(["timp_ore", "biomasa_g_L", "substrat_g_L", "oxigen_dizolvat_mg_L", "proteina_estimata_g_L"])
writer.writerows(zip(t, X, S, DO, P))

st.download_button(
    label="📥 Descarcă Datele Simulării (CSV)",
    data=csv_buffer.getvalue(),
    file_name="microprotein_simulation_results.csv",
    mime="text/csv"
)