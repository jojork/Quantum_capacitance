# Quantum_capacitance
Here's the script of finding the quantum capacitance
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Quantum Capacitance from DOS
#
# Cq(T, μ) = e^2 ∫ D(E) * [ -∂f(E, μ, T)/∂E ] dE
# where  -∂f/∂E = [1/(4 kB T)] * sech^2( (E-μ)/(2 kB T) )
#
# Units:
#   - Energy E in eV
#   - DOS D(E) in states/eV per cell
#   - Output Cq in Farads per cell
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import argparse
import os

# ----------------------------- Physical constants ----------------------------
e_charge = 1.602176634e-19     # Coulomb
kB_eV    = 8.617333262145e-5   # eV/K

# ------------------------------ DOS file parser ------------------------------
def parse_dos(path):
    energies, dos_total = [], []
    EF_header = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = re.search(r"Fermi level\s*=\s*([\-0-9.eE+]+)", line)
            if m:
                try:
                    EF_header = float(m.group(1))
                except:
                    pass
                continue

            s = line.replace("|", " ").replace(",", " ")
            low = s.lower()
            if any(h in low for h in ["energy", "dos", "density", "unit", "report"]):
                continue

            parts = s.split()
            try:
                vals = [float(p) for p in parts]
            except:
                continue

            if len(vals) >= 2:
                energies.append(vals[0])
                dos_total.append(sum(vals[1:]))

    E = np.array(energies)
    DOS = np.array(dos_total)

    idx = np.argsort(E)
    E, DOS = E[idx], DOS[idx]

    good = np.isfinite(E) & np.isfinite(DOS)
    return E[good], DOS[good], EF_header

# ---------------- Fermi–Dirac derivative  -df/dE -----------------------------
def minus_df_dE(E, mu, kBT):
    x = (E - mu) / (2.0 * kBT)
    return (1.0 / (4.0 * kBT)) / np.cosh(x)**2

# ---------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Quantum capacitance from DOS at finite temperature"
    )
    ap.add_argument(
    "--dosfile",
    type=str,
    default="~your DOS txt or csv file path",
    help="Path to DOS file (default: GPAW DOS file)")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--window", type=float, default=0.30)
    ap.add_argument("--outfile", type=str, default="cq_300K")
    ap.add_argument("--areaA2", type=float, default=None)
    args = ap.parse_args()

    dos_path = os.path.expanduser(args.dosfile)
    E, DOS, EF_header = parse_dos(dos_path)

    if E.size == 0:
        raise RuntimeError("DOS file contains no valid numeric data.")

    crosses_zero = (E.min() < 0) and (E.max() > 0)
    if not crosses_zero and EF_header is not None:
        E -= EF_header
        print(f"[info] Shifted energies by EF={EF_header:.6f} eV")
    else:
        print("[info] Energies appear already relative to EF (EF = 0 eV)")

    T = args.temp
    kBT = kB_eV * T
    print(f"[info] Temperature: {T:.1f} K  (kBT = {kBT:.5f} eV)")
    print(f"[info] Effective thermal window (±3kBT): ±{3*kBT:.3f} eV")

    w = args.window
    mask = (E >= -w) & (E <= w)
    Ew, DOSw = E[mask], DOS[mask]

    if Ew.size < 3:
        raise RuntimeError("DOS window too small around EF.")

    # -------------------- Cq at μ = 0 (EF) --------------------
    w_mu0 = minus_df_dE(Ew, 0.0, kBT)
    Cq_mu0 = (e_charge**2) * np.trapezoid(DOSw * w_mu0, Ew)
    print(f"Cq({T:.0f} K, μ=EF) = {Cq_mu0:.6e} F per cell")

    if args.areaA2:
        area_m2 = args.areaA2 * 1e-20
        print(f"Cq = {Cq_mu0/area_m2:.6e} F/m²")

    # -------------------- μ sweep ------------------------------
    mu_grid = np.linspace(-w, w, 301)
    Cq_vs_mu = np.array([
        (e_charge**2) * np.trapezoid(
            DOSw * minus_df_dE(Ew, mu, kBT), Ew
        )
        for mu in mu_grid
    ])

    # -------------------- Save CSV -----------------------------
    with open(f"{args.outfile}_Cq_vs_mu.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mu_eV", "Cq_F_per_cell"])
        wr.writerows(zip(mu_grid, Cq_vs_mu))

    with open(f"{args.outfile}_DOS_window.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Energy_eV", "DOS_states_per_eV"])
        wr.writerows(zip(Ew, DOSw))

    # -------------------- Plot -------------------------------
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(Ew, DOSw, label="DOS")
    ax1.axvline(0, ls="--", label="EF")
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("DOS")

    ax2 = ax1.twinx()
    ax2.plot(mu_grid, Cq_vs_mu, ls="--", label="Cq(μ)")
    ax2.set_ylabel("Quantum Capacitance (F)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)

    plt.tight_layout()
    plt.savefig(f"{args.outfile}_DOS_Cq_plot.png", dpi=180)
    plt.show()

# -------------------------------- entry --------------------------------------
if __name__ == "__main__":
    main()
