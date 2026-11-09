"""
SIMULATOR INTEGRAT DE MICROPROTEINE – model didactic de fermentatie batch
Titlu proiect: Simularea și optimizarea procesului de producție a micoproteinelor prin fermentație cu Fusarium venenatum
Autor: Jánosi-Rancz Júlia (Biotehnologii industriale I.)

Instalare:
    python -m pip install numpy scipy matplotlib

Rulare:
    python microprotein_integrat.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch
from scipy.integrate import solve_ivp

# Matplotlib saját mathtext engine-jét használjuk
plt.rcParams['text.usetex'] = False  

# ----------------------- PARAMETRI DIDACTICI -----------------------
DEFAULTS = {
    "mu_max": 0.20,       # 1/h, demonstrativ
    "T_opt": 30.0,        # °C, demonstrativ
    "T_sigma": 6.0,       # °C, lățimea curbei termice
    "Ks": 0.8,            # g/L
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
    
    # Radau solver a gyorsaságért és stabilitásért
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


def main():
    p0 = DEFAULTS.copy()
    
    # Configurare fereastră principală
    fig = plt.figure(figsize=(16, 9.5), facecolor="#f8f9fa")
    
    # Titlul principal al proiectului
    fig.suptitle(
        "Simularea și optimizarea procesului de producție a micoproteinelor prin fermentație cu Fusarium venenatum",
        fontsize=13, fontweight="bold", color="#1c7ed6", y=0.98
    )

    # GRAFICE (Posizionate cu spațiu între ele)
    ax_theory  = fig.add_axes([0.02, 0.28, 0.27, 0.65])
    ax_growth  = fig.add_axes([0.34, 0.65, 0.28, 0.26])
    ax_do      = fig.add_axes([0.34, 0.31, 0.28, 0.24])
    ax_sub     = fig.add_axes([0.705, 0.65, 0.27, 0.26])
    ax_compare = fig.add_axes([0.705, 0.31, 0.27, 0.24])

    ax_theory.axis("off")

    # Lekerekített hátterű doboz az egyenleteknek
    rect = FancyBboxPatch((0, 0), 1, 1, transform=ax_theory.transAxes,
                          facecolor='white', edgecolor='#ced4da', 
                          boxstyle="round,pad=0.03,rounding_size=0.03", alpha=0.95)
    ax_theory.add_patch(rect)

    # Cím a teóriához
    ax_theory.text(0.5, 0.96, "MODEL MATEMATIC (ECUAȚII)", fontsize=11, fontweight="bold", 
                   color="#1971c2", ha="center", va="top")

    # 1. Cinetica Monod Dublă
    ax_theory.text(0.05, 0.89, "1. Cinetică Monod Dublă:", fontsize=9, fontweight="bold", color="#212529")
    ax_theory.text(0.5, 0.82, r"$\mu = \mu_{\max} \cdot \gamma(T) \cdot \frac{S}{K_s + S} \cdot \frac{DO}{K_o + DO}$", 
                   fontsize=11, ha="center", va="center", color="#0b7285")

    # 2. Corecția Termică
    ax_theory.text(0.05, 0.74, "2. Corecția de Temperatură:", fontsize=9, fontweight="bold", color="#212529")
    ax_theory.text(0.5, 0.67, r"$\gamma(T) = \exp\left( -\left[ \frac{T - T_{\mathrm{opt}}}{T_{\sigma}} \right]^2 \right)$", 
                   fontsize=11, ha="center", va="center", color="#0b7285")

    # 3. Bilanțuri de Masă
    ax_theory.text(0.05, 0.59, "3. Bilanțuri de Masă (Diferențiale):", fontsize=9, fontweight="bold", color="#212529")
    ax_theory.text(0.08, 0.52, r"$\frac{dX}{dt} = \mu \cdot X$", fontsize=10.5, color="#2b8a3e")
    ax_theory.text(0.08, 0.44, r"$\frac{dS}{dt} = -\frac{1}{Y_{X/S}} \cdot \frac{dX}{dt}$", fontsize=10.5, color="#e65c00")
    ax_theory.text(0.08, 0.35, r"$\frac{dDO}{dt} = k_L a (C_{\mathrm{sat}} - DO) - 1000 \cdot q_{O2} \cdot X$", fontsize=10, color="#1971c2")

    # 4. Proteina Estimată
    ax_theory.text(0.05, 0.26, "4. Estimare Proteine:", fontsize=9, fontweight="bold", color="#212529")
    ax_theory.text(0.5, 0.20, r"$P = X \cdot \text{fracțiune\_proteică}$", fontsize=10.5, ha="center", va="center", color="#099268")

    # Parametri principali
    ax_theory.text(0.05, 0.12, "───────────────────────────────", fontsize=8, color="#adb5bd")
    ax_theory.text(0.05, 0.08, r"$\mu_{\max} = 0.20\,\mathrm{h^{-1}} \quad K_s = 0.8\,\mathrm{g/L}$", fontsize=8.5, color="#495057")
    ax_theory.text(0.05, 0.03, r"$Y_{X/S} = 0.48\,\mathrm{g/g} \quad K_o = 0.5\,\mathrm{mg/L}$", fontsize=8.5, color="#495057")

    # Nume autor stânga jos
    fig.text(
        0.02, 0.03,
        "Autor: Jánosi-Rancz Júlia\nSpecializare: Biotehnologii industriale I.",
        fontsize=9.5, fontweight="bold", color="#1971c2",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e7f5ff", edgecolor="#a5d8ff")
    )

    # SLIDERE
    slider_specs = [
        ("T", "Temperatură (°C)", 15, 40, 30),
        ("kLa", "Aerare kLa (1/h)", 1, 50, 15),
        ("S0", "Substrat inițial S_0 (g/L)", 5, 50, 25),
        ("mu_max", "μ_max (1/h)", 0.03, 0.35, 0.20),
        ("protein_frac", "Fracție proteică (g/g)", 0.20, 0.70, 0.45),
    ]
    sliders = {}
    ys = [0.20, 0.155, 0.11, 0.065, 0.02]
    for (key, label, lo, hi, val), y in zip(slider_specs, ys):
        ax_sl = fig.add_axes([0.42, y + 0.01, 0.34, 0.022])
        sliders[key] = Slider(ax_sl, label, lo, hi, valinit=val, color="#339af0")

    # GOMBOK
    reset_ax = fig.add_axes([0.80, 0.14, 0.15, 0.045])
    export_ax = fig.add_axes([0.80, 0.07, 0.15, 0.045])
    reset_btn = Button(reset_ax, "Resetare Parametri", color="#e9ecef", hovercolor="#dee2e6")
    export_btn = Button(export_ax, "Export Date (CSV)", color="#d0ebff", hovercolor="#a5d8ff")
    
    # STATUS BAR (ZÖLD EREDMÉNY SOR VISSZAÁLLÍTVA AZ ALSÓ SÁVBA)
    status_ax = fig.add_axes([0.34, 0.24, 0.635, 0.035])
    status_ax.axis("off")
    status_rect = FancyBboxPatch((0, 0), 1, 1, transform=status_ax.transAxes,
                                 facecolor='#e6fcf5', edgecolor='#20c997', 
                                 boxstyle="round,pad=0.02,rounding_size=0.02", alpha=0.9)
    status_ax.add_patch(status_rect)
    status_text = status_ax.text(0.5, 0.5, "", fontsize=9.5, fontweight="bold", color="#0ca678",
                                 ha="center", va="center")

    def current():
        p = p0.copy()
        for key, slider in sliders.items():
            p[key] = slider.val
        return p

    def draw(_=None):
        p = current()
        t, X, S, DO, P = simulate(p)

        # 1. Grafic Creștere Biomasă și Proteină
        ax_growth.clear()
        ax_growth.plot(t, X, label="Biomasă uscată (X)", color="#2b8a3e", lw=2.2)
        ax_growth.plot(t, P, label="Proteină estimată (P)", color="#099268", lw=2.2, linestyle="--")
        ax_growth.set_title("Creșterea Biomasei și Proteinei", fontsize=10, fontweight="bold")
        ax_growth.set_xlabel("Timp (h)", fontsize=8)
        ax_growth.set_ylabel("Concentrație (g/L)", fontsize=8)
        ax_growth.grid(alpha=0.3)
        ax_growth.legend(fontsize=8)

        # 2. Grafic Substrat
        ax_sub.clear()
        ax_sub.plot(t, S, color="#e65c00", lw=2.2, label="Substrat (S)")
        ax_sub.set_title("Consumul Substratului", fontsize=10, fontweight="bold")
        ax_sub.set_xlabel("Timp (h)", fontsize=8)
        ax_sub.set_ylabel("Substrat (g/L)", fontsize=8)
        ax_sub.grid(alpha=0.3)
        ax_sub.legend(fontsize=8)

        # 3. Grafic Oxigen Dizolvat
        ax_do.clear()
        ax_do.plot(t, DO, color="#1971c2", lw=2, label="Oxigen dizolvat (DO)")
        ax_do.axhline(p["Ko"], color="#c92a2a", ls=":", label="Nivel critic (Ko)")
        ax_do.set_title("Bilanțul Oxigenului Dizolvat", fontsize=10, fontweight="bold")
        ax_do.set_xlabel("Timp (h)", fontsize=8)
        ax_do.set_ylabel("DO (mg/L)", fontsize=8)
        ax_do.grid(alpha=0.3)
        ax_do.legend(fontsize=8)

        # 4. Grafic Comparație Scenarii S0
        ax_compare.clear()
        results = [summary(p, s) for s in SCENARIO_S0]
        max_p = max(r["Pf"] for r in results) if results else 1.0
        
        bars = ax_compare.bar(
            [f"{r['S0']:.0f}" for r in results],
            [r["Pf"] for r in results],
            color="#74c0fc"
        )
        ax_compare.set_title("Proteină finală la diferite S_0 inițiale", fontsize=10, fontweight="bold")
        ax_compare.set_xlabel("Substrat inițial S_0 (g/L)", fontsize=8)
        ax_compare.set_ylabel("Proteină (g/L)", fontsize=8)
        ax_compare.set_ylim(0, max_p * 1.25)
        ax_compare.grid(axis="y", alpha=0.3)
        
        for bar, r in zip(bars, results):
            ax_compare.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + (max_p * 0.02),
                f"{r['Pf']:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold"
            )

        # FRISSÍTETT ZÖLD EREDMÉNY SOR
        status_text.set_text(
            f"REZULTAT FINAL (t=30h): Biomasă (X) = {X[-1]:.2f} g/L  |  "
            f"Proteină (P) = {P[-1]:.2f} g/L  |  "
            f"Substrat Rămas (S) = {S[-1]:.2f} g/L"
        )
        fig.canvas.draw_idle()

    def reset(_event):
        for slider in sliders.values():
            slider.reset()
        draw()

    def export(_event):
        p = current()
        t, X, S, DO, P = simulate(p)
        filename = Path("microprotein_integrated_results.csv")
        with filename.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timp_ore", "biomasa_g_L", "substrat_g_L",
                "oxigen_dizolvat_mg_L", "proteina_estimata_g_L"
            ])
            writer.writerows(zip(t, X, S, DO, P))
        status_text.set_text(f"Fișier salvat cu succes: {filename.name}")
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(draw)
    reset_btn.on_clicked(reset)
    export_btn.on_clicked(export)

    draw()
    plt.show()


if __name__ == "__main__":
    main()