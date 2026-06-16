#!/usr/bin/env python3
"""Casi limite statici (corridoio stretto, ostacolo a U) per PF e DWA.

Uso:
    python3 scenari_statici.py                  # viewer + DWA sul corridoio
    python3 scenari_statici.py corridoio pf     # viewer + PF sul corridoio
    python3 scenari_statici.py u dwa            # viewer + DWA sull'ostacolo a U
    python3 scenari_statici.py --check          # verifica rapida headless
    python3 scenari_statici.py --compare        # tabella PF vs DWA su entrambi
"""

import contextlib
import io
import sys

import numpy as np

import potential_fields as pf
import dwa


# --------------------------------------------------------------------------
# Costruzione dei muri come sequenze di piccoli cilindri.
# --------------------------------------------------------------------------
WALL_R = 0.07
WALL_STEP = 0.20


def _wall_along_x(x_lo, x_hi, y, step=WALL_STEP, r=WALL_R):
    """Cilindri allineati lungo x al valore y dato (muro 'orizzontale')."""
    xs = np.arange(x_lo, x_hi + 1e-9, step)
    return [(float(x), float(y), float(r)) for x in xs]


def _wall_along_y(x, y_lo, y_hi, step=WALL_STEP, r=WALL_R):
    """Cilindri allineati lungo y al valore x dato (muro 'verticale')."""
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    return [(float(x), float(y), float(r)) for y in ys]


# --------------------------------------------------------------------------
# Scenari concreti
# --------------------------------------------------------------------------
CORRIDOIO_OBS = (
    _wall_along_x(1.5, 4.5, +0.55)
    + _wall_along_x(1.5, 4.5, -0.55)
)

U_OBS = (
    _wall_along_x(2.0, 3.0, +0.70)
    + _wall_along_x(2.0, 3.0, -0.70)
    + _wall_along_y(3.0, -0.70, +0.70)
)

SCENARI = {
    "corridoio": {
        "nome": "Corridoio stretto",
        "descrizione": "Due muri paralleli a y = +-0.55 m, x in [1.5, 4.5]: "
                       "tunnel libero di ~0.96 m.",
        "start": np.array([0.0, 0.0]),
        "goal": np.array([5.0, 0.0]),
        "obstacles": CORRIDOIO_OBS,
    },
    "u": {
        "nome": "Ostacolo a U",
        "descrizione": "U aperta verso il robot (apertura su -x, fondo su +x). "
                       "Classico minimo locale dei Potential Fields.",
        "start": np.array([0.0, 0.0]),
        "goal": np.array([5.0, 0.0]),
        "obstacles": U_OBS,
    },
}


# --------------------------------------------------------------------------
# Applicazione di uno scenario ai due moduli pf e dwa
# --------------------------------------------------------------------------
def applica_scenario(chiave):
    """Sostituisce START, GOAL e OBSTACLES in pf e dwa con lo scenario scelto."""
    s = SCENARI[chiave]
    pf.START = s["start"]
    pf.GOAL = s["goal"]
    pf.OBSTACLES = s["obstacles"]
    dwa.START = s["start"]
    dwa.GOAL = s["goal"]
    dwa.OBSTACLES = s["obstacles"]
    return s


def _run_silent(algoritmo):
    """Esegue pf.run() o dwa.run() headless silenziando il log per-step."""
    funzione = pf.run if algoritmo == "pf" else dwa.run
    with contextlib.redirect_stdout(io.StringIO()):
        return funzione(headless=True)


def run_scenario(chiave, algoritmo, headless=False):
    """Applica lo scenario e lancia l'algoritmo con stampe attive."""
    applica_scenario(chiave)
    funzione = pf.run if algoritmo == "pf" else dwa.run
    return funzione(headless=headless)


# --------------------------------------------------------------------------
# Confronto: PF vs DWA su entrambi gli scenari
# --------------------------------------------------------------------------
def _stampa_tabella(m_pf, m_dwa):
    """Stampa la tabella PF vs DWA per uno scenario."""
    righe = [
        ("Esito",                  "esito",          None),
        ("Step",                   "step",           "d"),
        ("Tempo simulato [s]",     "tempo_s",        ".2f"),
        ("Distanza finale [m]",    "dist_finale",    ".3f"),
        ("Lunghezza percorso [m]", "lunghezza_path", ".3f"),
        ("Clearance minima [m]",   "clearance_min",  ".3f"),
    ]
    print(f"{'METRICA':<24} | {'Potential Fields':>18} | {'DWA':>18}")
    print("-" * 68)
    for label, key, fmt in righe:
        a, b = m_pf[key], m_dwa[key]
        if fmt is None:
            print(f"{label:<24} | {str(a):>18} | {str(b):>18}")
        else:
            print(f"{label:<24} | {a:>18{fmt}} | {b:>18{fmt}}")
    print("-" * 68)


def _verdetto(nome, m_pf, m_dwa):
    """Verdetto testuale del confronto: 4 casi (entrambi/PF/DWA/nessuno al goal)."""
    pf_ok = m_pf["esito"] == "goal_raggiunto"
    dwa_ok = m_dwa["esito"] == "goal_raggiunto"
    if pf_ok and dwa_ok:
        return f"{nome}: entrambi gli algoritmi raggiungono il goal."
    if pf_ok:
        return f"{nome}: SOLO Potential Fields raggiunge il goal (DWA: {m_dwa['esito']})."
    if dwa_ok:
        return f"{nome}: SOLO DWA raggiunge il goal (PF: {m_pf['esito']})."
    return (f"{nome}: ENTRAMBI falliscono "
            f"(PF: {m_pf['esito']}, DWA: {m_dwa['esito']}).")


def confronta():
    """Esegue PF e DWA su tutti gli scenari e stampa tabelle + verdetti."""
    print("=" * 92)
    print("SCENARI STATICI - 'casi limite' - Potential Fields vs DWA")
    print("=" * 92)

    risultati = {}
    for chiave, scenario in SCENARI.items():
        print()
        print(f">>> Scenario: {scenario['nome']}")
        print(f"    {scenario['descrizione']}")
        print(f"    Start = ({scenario['start'][0]:.1f}, {scenario['start'][1]:.1f})   "
              f"Goal  = ({scenario['goal'][0]:.1f}, {scenario['goal'][1]:.1f})   "
              f"# cilindri = {len(scenario['obstacles'])}")
        print("    esecuzione Potential Fields (headless)...", end=" ", flush=True)
        applica_scenario(chiave)
        m_pf = _run_silent("pf")
        print(m_pf["esito"])
        print("    esecuzione DWA (headless)................", end=" ", flush=True)
        applica_scenario(chiave)
        m_dwa = _run_silent("dwa")
        print(m_dwa["esito"])
        risultati[chiave] = (m_pf, m_dwa)

    print()
    print("=" * 92)
    print("RISULTATI")
    print("=" * 92)
    for chiave, (m_pf, m_dwa) in risultati.items():
        print()
        print(f"--- {SCENARI[chiave]['nome']} ---")
        _stampa_tabella(m_pf, m_dwa)

    print()
    print("=" * 92)
    print("VERDETTO")
    print("=" * 92)
    for chiave, (m_pf, m_dwa) in risultati.items():
        print("  " + _verdetto(SCENARI[chiave]["nome"], m_pf, m_dwa))


# --------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]

    if "--check" in args:
        for chiave in SCENARI:
            for alg in ("pf", "dwa"):
                applica_scenario(chiave)
                with contextlib.redirect_stdout(io.StringIO()):
                    m = (pf.run if alg == "pf" else dwa.run)(headless=True)
                print(f"  {SCENARI[chiave]['nome']:<22} {alg.upper():>4}: "
                      f"esito={m['esito']:<16} step={m['step']:>4}  "
                      f"dist_finale={m['dist_finale']:.3f}")
        return

    if "--compare" in args:
        confronta()
        return

    chiave = "corridoio"
    algoritmo = "dwa"
    for a in args:
        if a in SCENARI:
            chiave = a
        elif a in ("pf", "dwa"):
            algoritmo = a
    print(f"Scenario: {SCENARI[chiave]['nome']}   Algoritmo: {algoritmo.upper()}")
    run_scenario(chiave, algoritmo, headless=False)


if __name__ == "__main__":
    main()
