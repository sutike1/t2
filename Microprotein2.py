"""
SIMULATOR INTEGRAT DE MICOPROTEINE – model didactic de fermentatie batch
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
       .highlight-word {
            color: #ff4b4b !important;
            font-weight: bold;
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
    "T": 30.0,            # °C, valoarea inițială a sliderului
    "T_opt": 30.0,        # °C, demonstrativ
    "T_sigma": 6.0,       # °C, lățimea curbei termice
    "Ks": 0.0054,         # g/L (F. venenatum glükóz affinitása: 30 µM ≈ 5.4 mg/L)
    "Ko": 0.5,            # mg O2/L, constanta de semisaturație pentru oxigen
    "Csat": 8.0,          # mg O2/L
    "kLa": 150.0,         # 1/h
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


def simulate(p, S0=None, kLa=None, points=200):
    p_sim = p.copy()
    if S0 is not None:
        p_sim["S0"] = float(S0)
    if kLa is not None:
        p_sim["kLa"] = float(kLa)
    t_eval = np.linspace(0, p_sim["hours"], points)
    y0 = [p_sim["X0"], p_sim["S0"], p_sim["DO0"]]
    
    sol = solve_ivp(
        lambda t, y: rhs(t, y, p_sim),
        (0, p_sim["hours"]), y0, t_eval=t_eval,
        method="LSODA", rtol=1e-5, atol=1e-7
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


def diagnose(p, t, X, S, DO):
    """Care factor a determinat rezultatul final?"""
    gamma_T = float(np.exp(-((p["T"] - p["T_opt"]) / p["T_sigma"]) ** 2))
    fO = DO / (p["Ko"] + DO)
    w = np.maximum(np.gradient(X, t), 0.0)
    fO_mean = float(np.average(fO, weights=w)) if w.sum() > 1e-9 else float(np.mean(fO))

    if S[-1] <= 0.01 * p["S0"]:
        kind = "substrat"
    elif min(gamma_T, fO_mean) >= 0.9:
        kind = "timp"
    elif gamma_T <= fO_mean:
        kind = "temperatura"
    else:
        kind = "oxigen"
    return kind, gamma_T, fO_mean


@st.cache_data(show_spinner=False)
def cached_protein(base_key, T, kLa, S0):
    """Proteina finală (g/L) pentru o combinație T, kLa, S_0; base_key = restul parametrilor."""
    q = dict(base_key)
    q["T"] = float(T)
    _, _, _, _, Pq = simulate(q, S0=float(S0), kLa=float(kLa), points=2)
    return float(Pq[-1])


# ----------------------- STREAMLIT UI -----------------------
st.markdown(
    '<h1>Simularea procesului de producție de <span class="highlight-word">micoproteină</span> '
    'prin fermentație cu <span class="highlight-word">Fusarium venenatum</span></h1>',
    unsafe_allow_html=True
)
st.markdown("**Autor:** Jánosi-Rancz Júlia | **Specializare:** Biotehnologii industriale I.")

# Valorile inițiale ale sliderelor (session_state = singura sursă de adevăr)
SLIDER_DEFAULTS = {
    "T": DEFAULTS["T"],
    "kLa": DEFAULTS["kLa"],
    "S0": DEFAULTS["S0"],
    "mu_max": DEFAULTS["mu_max"],
    "protein_frac": DEFAULTS["protein_frac"],
}
for k, v in SLIDER_DEFAULTS.items():
    st.session_state.setdefault(k, v)

# BAL OLDALI SÁV (SLIDEREK)
st.sidebar.header("Parametrii de simulare")
if st.sidebar.button("🔄 Resetare parametrii"):
    for k, v in SLIDER_DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

p = DEFAULTS.copy()
p["T"] = st.sidebar.slider("Temperatură (°C)", 15.0, 40.0, key="T")
p["kLa"] = st.sidebar.slider("Aerare kLa (1/h)", 10.0, 400.0, key="kLa")
p["S0"] = st.sidebar.slider("Substrat inițial S_0 (g/L)", 5.0, 50.0, key="S0")
p["mu_max"] = st.sidebar.slider("μ_max (1/h)", 0.03, 0.35, key="mu_max")
p["protein_frac"] = st.sidebar.slider("Fracție proteică (g/g)", 0.20, 0.70, key="protein_frac")

# PARAMETRII INIȚIALI KIÍRÁSA A CSÚSZKÁK ALATT
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Parametrii inițiali")
st.sidebar.markdown(f"""
* **Temperatură ($T_0$):** `{DEFAULTS['T']:.1f} °C` 
* **Biomasă inițială ($X_0$):** `{DEFAULTS['X0']:.2f} g/L`
* **Substrat inițial ($S_0$):** `{DEFAULTS['S0']:.1f} g/L`
* **Oxigen dizolvat ($DO_0$):** `{DEFAULTS['DO0']:.1f} mg/L`
* **Viteza max. de creștere ($\mu_{{\max}}$):** `{DEFAULTS['mu_max']:.2f} 1/h`
* **Aerare ($k_L a$):** `{DEFAULTS['kLa']:.1f} 1/h`
* **Constatantă $K_s$ (glucoză):** `{DEFAULTS['Ks']*1000:.1f} mg/L` 
* **Constatantă $K_o$ (oxigen):** `{DEFAULTS['Ko']:.1f} mg O₂/L`
* **Oxigen la saturație ($C_{{\mathrm{{sat}}}}$):** `{DEFAULTS['Csat']:.1f} mg/L`
* **Randament biomasă ($Y_{{X/S}}$):** `{DEFAULTS['Yxs']:.2f} g/g`
* **Consum specific $O_2$ ($q_{{O2}}$):** `{DEFAULTS['qO2']:.2f} g O₂/(g·h)`
* **Fracție proteică:** `{DEFAULTS['protein_frac']*100:.0f}%` (`{DEFAULTS['protein_frac']:.2f} g/g`)
* **Durată fermentație:** `{DEFAULTS['hours']:.0f} h`
""")

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


# 1. GRAFIKONOK KIRAJZOLÁSA (ELŐRE HOZVA)
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
    plt.close(fig1)

    fig2, ax_do = plt.subplots(figsize=(5, 2.2))
    ax_do.plot(t, DO, color="#1971c2", lw=2, label="Oxigen dizolvat (DO)")
    ax_do.axhline(p["Ko"], color="#c92a2a", ls=":", label="Const. de semisaturație (Ko)")
    ax_do.set_title("Bilanțul Oxigenului Dizolvat", fontsize=9, fontweight="bold")
    ax_do.set_xlabel("Timp (h)", fontsize=7.5)
    ax_do.set_ylabel("DO (mg/L)", fontsize=7.5)
    ax_do.tick_params(labelsize=7)
    ax_do.grid(alpha=0.3)
    ax_do.legend(fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

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
    plt.close(fig3)

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
    plt.close(fig4)


# 2. EREDMÉNY KIÍRÁSA ZÖLD KÁRTYÁBAN (GRAFIKONOK UTÁN)
st.success(
    f"**REZULTAT FINAL (t={p['hours']:.0f}h):** "
    f"Biomasă (X) = **{X[-1]:.2f} g/L** | "
    f"Proteină (P) = **{P[-1]:.2f} g/L** | "
    f"Substrat Rămas (S) = **{S[-1]:.2f} g/L**"
)

# 3. FACTORUL LIMITANT (ZÖLD KÁRTYA UTÁN)
kind, gamma_T, fO_mean = diagnose(p, t, X, S, DO)
if kind == "substrat":
    st.info(
        f"**Factor limitant: SUBSTRATUL.** Glucoza s-a epuizat înainte de finalul "
        f"fermentației (S final ≈ {S[-1]:.2f} g/L). Pentru mai multă proteină mărește S_0 "
        f"(și, la nevoie, kLa)."
    )
elif kind == "oxigen":
    st.info(
        f"**Factor limitant: OXIGENUL.** Au rămas {S[-1]:.1f} g/L de substrat nefolosit, dar "
        f"aprovizionarea cu oxigen nu ține pasul cu consumul. Pentru mai multă proteină mărește kLa."
    )
elif kind == "temperatura":
    st.info(
        f"**Factor limitant: TEMPERATURA.** Au rămas {S[-1]:.1f} g/L de substrat nefolosit; "
        f"la T = {p['T']:.1f} °C viteza de creștere este redusă la {gamma_T * 100:.0f}% din maxim. "
        f"Apropie T de T_opt = {p['T_opt']:.0f} °C."
    )
else:
    st.info(
        f"**Factor limitant: TIMPUL.** Nici oxigenul, nici temperatura nu limitează, dar în "
        f"{p['hours']:.0f} h substratul nu se epuizează (rămân {S[-1]:.1f} g/L). "
        f"Mărește μ_max sau reduce S_0."
    )


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