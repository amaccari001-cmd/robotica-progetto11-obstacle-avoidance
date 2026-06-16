#!/usr/bin/env python3
"""Genera i grafici matplotlib del progetto con cache su disco.

Output:
  dati/metriche.csv       - una riga per (scenario, algoritmo) con le metriche
  dati/traiettorie.json   - sequenze x,y di ogni simulazione
  grafici/*.png           - 5 grafici matplotlib

Uso:
    python3 genera_grafici.py             # carica da cache se possibile
    python3 genera_grafici.py --rigenera  # forza il re-run delle simulazioni
"""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import potential_fields as pf
import dwa
import scenari_statici as scen
import dynamic_obstacles as dyn

DT = pf.DT
V_MAX = pf.V_MAX
GOAL_TOL = pf.GOAL_TOL
MAX_STEPS = pf.MAX_STEPS
STUCK_STEPS = pf.STUCK_STEPS
ROBOT_RADIUS = dwa.ROBOT_RADIUS

SLALOM_OBS = list(pf.OBSTACLES)

ROOT = Path(__file__).resolve().parent
DATI_DIR = ROOT / "dati"
METRICHE_FILE = DATI_DIR / "metriche.csv"
TRAIETTORIE_FILE = DATI_DIR / "traiettorie.json"
OUT_DIR = ROOT / "grafici"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAMPI_METRICHE = ["scenario", "algoritmo", "esito", "step", "lunghezza_path",
                  "clearance_min", "collision_steps", "dist_finale"]
CAMPI_INT = {"step", "collision_steps"}
CAMPI_FLOAT = {"lunghezza_path", "clearance_min", "dist_finale"}

COL_PF = "#1f77b4"
COL_DWA = "#ff7f0e"
COL_OBS = "#7f7f7f"
COL_START = "#2ca02c"
COL_GOAL = "#d62728"
COL_OBS_MOB = ["#ff9900", "#9933cc"]


# --------------------------------------------------------------------------
# Simulazioni: registrano traiettorie e clearance minima.
# --------------------------------------------------------------------------
def _clearance(pos, obs_list):
    """Distanza minima dal robot (punto) alla superficie degli ostacoli."""
    return min(math.hypot(pos[0] - ox, pos[1] - oy) - r
               for (ox, oy, r) in obs_list)


def simula_pf(start, goal, obstacles_iniziali, dyn_fn=None):
    """Ritorna (traj_robot, traj_obstacles, metriche). dyn_fn opzionale: t -> obs."""
    pf.START = np.asarray(start, dtype=float)
    pf.GOAL = np.asarray(goal, dtype=float)
    pf.OBSTACLES = list(obstacles_iniziali)

    pos = np.asarray(start, dtype=float).copy()
    traj_robot = [pos.copy()]
    traj_obs = [list(pf.OBSTACLES)]
    stuck = 0
    collision_steps = 0
    clearance_min = _clearance(pos, pf.OBSTACLES)
    esito = "max_steps"
    path_length = 0.0
    step = 0

    for step in range(MAX_STEPS):
        if dyn_fn is not None:
            pf.OBSTACLES = dyn_fn(step * DT)

        f_att = pf.attractive_force(pos)
        f_rep = pf.repulsive_force(pos)
        f_tot = f_att + f_rep
        speed = float(np.linalg.norm(f_tot))
        vel = f_tot / speed * V_MAX if speed > V_MAX else f_tot

        prev = pos
        pos = pos + vel * DT
        path_length += float(np.linalg.norm(pos - prev))
        traj_robot.append(pos.copy())
        traj_obs.append(list(pf.OBSTACLES))

        clr = _clearance(pos, pf.OBSTACLES)
        clearance_min = min(clearance_min, clr)
        if clr < ROBOT_RADIUS:
            collision_steps += 1

        dist = float(np.linalg.norm(pf.GOAL - pos))
        if dist < GOAL_TOL:
            esito = "goal"
            break
        stuck = stuck + 1 if np.linalg.norm(vel) < 0.02 else 0
        if stuck >= STUCK_STEPS:
            esito = "min. locale"
            break

    return (np.array(traj_robot), traj_obs,
            {"algoritmo": "PF", "esito": esito, "step": step,
             "lunghezza_path": path_length, "clearance_min": clearance_min,
             "collision_steps": collision_steps,
             "dist_finale": float(np.linalg.norm(pf.GOAL - pos))})


def simula_dwa(start, goal, obstacles_iniziali, dyn_fn=None):
    """Ritorna (traj_robot, traj_obstacles, metriche). dyn_fn opzionale: t -> obs."""
    dwa.START = np.asarray(start, dtype=float)
    dwa.GOAL = np.asarray(goal, dtype=float)
    dwa.OBSTACLES = list(obstacles_iniziali)

    state = np.array([start[0], start[1], 0.0])
    v, omega = 0.0, 0.0
    traj_robot = [state[:2].copy()]
    traj_obs = [list(dwa.OBSTACLES)]
    stuck = 0
    collision_steps = 0
    clearance_min = _clearance(state[:2], dwa.OBSTACLES)
    esito = "max_steps"
    path_length = 0.0
    step = 0

    for step in range(MAX_STEPS):
        if dyn_fn is not None:
            dwa.OBSTACLES = dyn_fn(step * DT)

        v, omega, _ = dwa.plan(state, v, omega)
        prev_xy = state[:2].copy()
        state = dwa.motion(state, v, omega, DT)
        state[2] = math.atan2(math.sin(state[2]), math.cos(state[2]))
        path_length += float(np.linalg.norm(state[:2] - prev_xy))
        traj_robot.append(state[:2].copy())
        traj_obs.append(list(dwa.OBSTACLES))

        clr = _clearance(state[:2], dwa.OBSTACLES)
        clearance_min = min(clearance_min, clr)
        if clr < ROBOT_RADIUS:
            collision_steps += 1

        dist = float(np.linalg.norm(dwa.GOAL - state[:2]))
        if dist < GOAL_TOL:
            esito = "goal"
            break
        stuck = stuck + 1 if v < 0.02 else 0
        if stuck >= STUCK_STEPS:
            esito = "stallo"
            break

    return (np.array(traj_robot), traj_obs,
            {"algoritmo": "DWA", "esito": esito, "step": step,
             "lunghezza_path": path_length, "clearance_min": clearance_min,
             "collision_steps": collision_steps,
             "dist_finale": float(np.linalg.norm(dwa.GOAL - state[:2]))})


# --------------------------------------------------------------------------
# Persistenza
# --------------------------------------------------------------------------
def _to_jsonable(arr):
    """Converte una traiettoria numpy in lista di liste di float arrotondati."""
    return np.round(np.asarray(arr), 5).tolist()


def _pack_scenario(start, goal, obs_iniziali, is_dinamico,
                   traj_pf, obs_pf, m_pf,
                   traj_dwa, obs_dwa, m_dwa):
    """Costruisce la struttura cachabile per uno scenario."""
    def obs_traj_serializable(obs_list_per_step):
        if obs_list_per_step is None:
            return None
        return [[[round(float(c), 5) for c in ostacolo] for ostacolo in snap]
                for snap in obs_list_per_step]
    return {
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "obstacles_iniziali": [[float(o[0]), float(o[1]), float(o[2])]
                               for o in obs_iniziali],
        "is_dinamico": bool(is_dinamico),
        "pf": {
            "traj_robot": _to_jsonable(traj_pf),
            "traj_obstacles": obs_traj_serializable(obs_pf),
            "metriche": m_pf,
        },
        "dwa": {
            "traj_robot": _to_jsonable(traj_dwa),
            "traj_obstacles": obs_traj_serializable(obs_dwa),
            "metriche": m_dwa,
        },
    }


def esegui_simulazioni():
    """Esegue le 8 simulazioni (4 scenari x 2 algoritmi) e ritorna il dict."""
    print("Eseguo le simulazioni headless...")
    risultati = {}

    print("  scenario 1/4: slalom")
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 0.0])
    traj_pf, _, m_pf = simula_pf(start, goal, SLALOM_OBS)
    traj_dwa, _, m_dwa = simula_dwa(start, goal, SLALOM_OBS)
    risultati["slalom"] = _pack_scenario(start, goal, SLALOM_OBS, False,
                                         traj_pf, None, m_pf,
                                         traj_dwa, None, m_dwa)

    print("  scenario 2/4: corridoio")
    s = scen.SCENARI["corridoio"]
    traj_pf, _, m_pf = simula_pf(s["start"], s["goal"], s["obstacles"])
    traj_dwa, _, m_dwa = simula_dwa(s["start"], s["goal"], s["obstacles"])
    risultati["corridoio"] = _pack_scenario(s["start"], s["goal"], s["obstacles"],
                                            False,
                                            traj_pf, None, m_pf,
                                            traj_dwa, None, m_dwa)

    print("  scenario 3/4: ostacolo a U")
    s = scen.SCENARI["u"]
    traj_pf, _, m_pf = simula_pf(s["start"], s["goal"], s["obstacles"])
    traj_dwa, _, m_dwa = simula_dwa(s["start"], s["goal"], s["obstacles"])
    risultati["u"] = _pack_scenario(s["start"], s["goal"], s["obstacles"], False,
                                    traj_pf, None, m_pf,
                                    traj_dwa, None, m_dwa)

    print("  scenario 4/4: dinamico (fase 0)")
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 0.0])
    dyn_fn = dyn.dynamic_obstacles
    obs_iniziali = dyn_fn(0.0)
    traj_pf, obs_pf, m_pf = simula_pf(start, goal, obs_iniziali, dyn_fn=dyn_fn)
    traj_dwa, obs_dwa, m_dwa = simula_dwa(start, goal, obs_iniziali, dyn_fn=dyn_fn)
    risultati["dinamico"] = _pack_scenario(start, goal, obs_iniziali, True,
                                           traj_pf, obs_pf, m_pf,
                                           traj_dwa, obs_dwa, m_dwa)

    return risultati


def salva_dati(risultati):
    """Sovrascrive metriche.csv e traiettorie.json."""
    DATI_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(METRICHE_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPI_METRICHE)
        w.writeheader()
        for nome_scenario, dati in risultati.items():
            for alg in ("pf", "dwa"):
                m = dati[alg]["metriche"]
                row = {"scenario": nome_scenario}
                for k in CAMPI_METRICHE[1:]:
                    v = m[k]
                    row[k] = f"{v:.6f}" if k in CAMPI_FLOAT else v
                w.writerow(row)
    print(f"  dati/metriche.csv aggiornato il {now}")

    payload = {
        "metadata": {"generato_il": now, "dt": DT,
                     "info": "traiettorie e metriche di genera_grafici.py"},
        "simulazioni": risultati,
    }
    with open(TRAIETTORIE_FILE, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"  dati/traiettorie.json aggiornato il {now}")


def carica_dati():
    """Carica dai file di cache. Solleva su errori (gestiti dal chiamante)."""
    with open(TRAIETTORIE_FILE) as f:
        payload = json.load(f)
    risultati = payload["simulazioni"]

    with open(METRICHE_FILE) as f:
        for row in csv.DictReader(f):
            sc = row["scenario"]
            alg = row["algoritmo"].lower()
            m = dict(row)
            del m["scenario"]
            for k in CAMPI_INT:
                m[k] = int(m[k])
            for k in CAMPI_FLOAT:
                m[k] = float(m[k])
            risultati[sc][alg]["metriche"] = m

    mtime_m = datetime.fromtimestamp(METRICHE_FILE.stat().st_mtime
                                     ).strftime("%Y-%m-%d %H:%M:%S")
    mtime_j = datetime.fromtimestamp(TRAIETTORIE_FILE.stat().st_mtime
                                     ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  dati/metriche.csv caricato (ultimo aggiornamento: {mtime_m})")
    print(f"  dati/traiettorie.json caricato (ultimo aggiornamento: {mtime_j})")
    return risultati


def carica_o_simula(rigenera=False):
    """Carica dalla cache se possibile, altrimenti simula e salva."""
    if rigenera:
        print("Flag --rigenera: rieseguo tutte le simulazioni.")
    elif METRICHE_FILE.exists() and TRAIETTORIE_FILE.exists():
        try:
            print("Cache trovata, caricamento da disco...")
            return carica_dati()
        except (json.JSONDecodeError, KeyError, ValueError, csv.Error) as e:
            print(f"  Cache corrotta ({e!r}), procedo a rigenerare.")
    else:
        print("Cache non trovata.")
    risultati = esegui_simulazioni()
    salva_dati(risultati)
    return risultati


# --------------------------------------------------------------------------
# Helper di plotting
# --------------------------------------------------------------------------
def _disegna_ostacoli(ax, obstacles, color=COL_OBS, alpha=0.7, label="Ostacolo"):
    """Cerchi grigi (Circle in coordinate dati) per gli ostacoli."""
    for i, (ox, oy, r) in enumerate(obstacles):
        ax.add_patch(patches.Circle((ox, oy), r, facecolor=color,
                                    edgecolor="black", alpha=alpha, linewidth=0.5,
                                    label=label if i == 0 else None))


def _disegna_start_goal(ax, start, goal):
    """Marker di partenza (cerchio verde) e goal (stella rossa)."""
    ax.plot(start[0], start[1], marker="o", color=COL_START, markersize=11,
            markeredgecolor="black", zorder=5, label="Start", linestyle="None")
    ax.plot(goal[0], goal[1], marker="*", color=COL_GOAL, markersize=18,
            markeredgecolor="black", zorder=5, label="Goal", linestyle="None")


def _figura_traiettorie(start, goal, obs, traj_pf, traj_dwa, m_pf, m_dwa,
                        titolo, filename, xlim=None, ylim=None):
    """Pannello unico con traiettorie PF + DWA, ostacoli e marker."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _disegna_ostacoli(ax, obs)
    _disegna_start_goal(ax, start, goal)
    ax.plot(traj_pf[:, 0], traj_pf[:, 1], color=COL_PF, linewidth=2.2,
            label=f"PF ({m_pf['esito']}, {m_pf['step']} step, "
                  f"L={m_pf['lunghezza_path']:.2f} m)")
    ax.plot(traj_dwa[:, 0], traj_dwa[:, 1], color=COL_DWA, linewidth=2.2,
            linestyle="--",
            label=f"DWA ({m_dwa['esito']}, {m_dwa['step']} step, "
                  f"L={m_dwa['lunghezza_path']:.2f} m)")
    ax.plot(traj_pf[-1, 0], traj_pf[-1, 1], marker="X", color=COL_PF,
            markersize=11, markeredgecolor="black", zorder=6, linestyle="None")
    ax.plot(traj_dwa[-1, 0], traj_dwa[-1, 1], marker="X", color=COL_DWA,
            markersize=11, markeredgecolor="black", zorder=6, linestyle="None")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(titolo)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=130)
    plt.close(fig)
    print(f"  salvato: grafici/{filename}")


# --------------------------------------------------------------------------
# Funzioni di plotting: ricevono i dati gia' pronti.
# --------------------------------------------------------------------------
def _estrai(dati):
    """Destruttura il dict di uno scenario nelle 7 variabili usate dai plot."""
    return (np.array(dati["start"]),
            np.array(dati["goal"]),
            dati["obstacles_iniziali"],
            np.array(dati["pf"]["traj_robot"]),
            np.array(dati["dwa"]["traj_robot"]),
            dati["pf"]["metriche"],
            dati["dwa"]["metriche"])


def grafico_slalom(dati):
    """Grafico delle traiettorie nello scenario slalom."""
    print("Grafico 1/5: slalom")
    start, goal, obs, traj_pf, traj_dwa, m_pf, m_dwa = _estrai(dati)
    _figura_traiettorie(start, goal, obs, traj_pf, traj_dwa, m_pf, m_dwa,
                        "Scenario base: slalom (3 ostacoli statici)",
                        "traiettorie_slalom.png",
                        xlim=(-0.4, 5.6), ylim=(-1.8, 1.4))


def grafico_corridoio(dati):
    """Grafico delle traiettorie nel corridoio stretto."""
    print("Grafico 2/5: corridoio stretto")
    start, goal, obs, traj_pf, traj_dwa, m_pf, m_dwa = _estrai(dati)
    _figura_traiettorie(start, goal, obs, traj_pf, traj_dwa, m_pf, m_dwa,
                        "Corridoio stretto: due muri paralleli a y = ±0.55 m",
                        "traiettorie_corridoio.png",
                        xlim=(-0.4, 5.6), ylim=(-1.1, 1.1))


def grafico_u(dati):
    """Grafico delle traiettorie nello scenario a U."""
    print("Grafico 3/5: ostacolo a U")
    start, goal, obs, traj_pf, traj_dwa, m_pf, m_dwa = _estrai(dati)
    _figura_traiettorie(start, goal, obs, traj_pf, traj_dwa, m_pf, m_dwa,
                        "Ostacolo a U (apertura verso il robot)",
                        "traiettorie_u.png",
                        xlim=(-0.4, 5.6), ylim=(-1.3, 1.3))


def _pannello_dinamico(ax, traj_robot, traj_obs, start, goal, m, titolo,
                       col_robot):
    """Singolo pannello del grafico dinamico (uno per PF, uno per DWA)."""
    nomi_obs = ["lineare", "circolare"]
    for i in range(len(traj_obs[0])):
        xs = [snap[i][0] for snap in traj_obs]
        ys = [snap[i][1] for snap in traj_obs]
        ax.plot(xs, ys, linestyle=":", color=COL_OBS_MOB[i], alpha=0.55,
                linewidth=1.2, label=f"traiettoria ostacolo {nomi_obs[i]}")
        ix, iy, ir = traj_obs[0][i]
        fx, fy, fr = traj_obs[-1][i]
        ax.add_patch(patches.Circle((ix, iy), ir, facecolor=COL_OBS_MOB[i],
                                    edgecolor="black", alpha=0.25, linewidth=0.5))
        ax.add_patch(patches.Circle((fx, fy), fr, facecolor=COL_OBS_MOB[i],
                                    edgecolor="black", alpha=0.85, linewidth=0.5,
                                    label=f"ostacolo {nomi_obs[i]} (fine)"))
    _disegna_start_goal(ax, start, goal)
    ax.plot(traj_robot[:, 0], traj_robot[:, 1], color=col_robot, linewidth=2.4,
            label=f"Robot ({m['esito']}, {m['step']} step, "
                  f"{m['collision_steps']} in collisione)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_title(titolo)
    ax.set_xlim(-0.4, 5.6)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)


def grafico_dinamico(dati):
    """Due pannelli affiancati (PF | DWA) per lo scenario dinamico."""
    print("Grafico 4/5: dinamico")
    start = np.array(dati["start"])
    goal = np.array(dati["goal"])
    traj_pf = np.array(dati["pf"]["traj_robot"])
    traj_dwa = np.array(dati["dwa"]["traj_robot"])
    obs_pf = dati["pf"]["traj_obstacles"]
    obs_dwa = dati["dwa"]["traj_obstacles"]
    m_pf = dati["pf"]["metriche"]
    m_dwa = dati["dwa"]["metriche"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    _pannello_dinamico(axes[0], traj_pf, obs_pf, start, goal, m_pf,
                       "Potential Fields", COL_PF)
    _pannello_dinamico(axes[1], traj_dwa, obs_dwa, start, goal, m_dwa,
                       "DWA", COL_DWA)
    axes[0].set_ylabel("y [m]")
    fig.suptitle("Scenario dinamico (2 ostacoli mobili, fase iniziale 0): "
                 "Potential Fields vs DWA", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "traiettorie_dinamico.png", dpi=130)
    plt.close(fig)
    print("  salvato: grafici/traiettorie_dinamico.png")


def grafico_metriche(risultati):
    """Bar plot riassuntivo delle 3 metriche chiave per tutti gli scenari."""
    print("Grafico 5/5: bar plot metriche")
    scenari_ordinati = [("slalom", "Slalom"), ("corridoio", "Corridoio"),
                        ("u", "Ostacolo a U"), ("dinamico", "Dinamico")]
    metriche = [
        ("step",            "Step (al goal o al blocco)"),
        ("lunghezza_path",  "Lunghezza percorso [m]"),
        ("collision_steps", "Step in collisione"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(scenari_ordinati))
    width = 0.38
    for ax, (key, titolo) in zip(axes, metriche):
        ypf = [risultati[k]["pf"]["metriche"][key] for k, _ in scenari_ordinati]
        ydwa = [risultati[k]["dwa"]["metriche"][key] for k, _ in scenari_ordinati]
        b1 = ax.bar(x - width/2, ypf, width, color=COL_PF, label="PF",
                    edgecolor="black", linewidth=0.5)
        b2 = ax.bar(x + width/2, ydwa, width, color=COL_DWA, label="DWA",
                    edgecolor="black", linewidth=0.5)
        ymax = max(max(ypf), max(ydwa), 1)
        for bar, v in list(zip(b1, ypf)) + list(zip(b2, ydwa)):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02 * ymax,
                    f"{v:.0f}" if key != "lunghezza_path" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([nome for _, nome in scenari_ordinati], rotation=10)
        ax.set_ylabel(titolo)
        ax.set_title(titolo)
        ax.set_ylim(top=ymax * 1.28)
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.legend(loc="upper left")
    fig.suptitle("Potential Fields vs DWA - metriche per scenario", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metriche_confronto.png", dpi=130)
    plt.close(fig)
    print("  salvato: grafici/metriche_confronto.png")


# --------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(line_buffering=True)
    rigenera = "--rigenera" in sys.argv

    risultati = carica_o_simula(rigenera=rigenera)

    print()
    grafico_slalom(risultati["slalom"])
    grafico_corridoio(risultati["corridoio"])
    grafico_u(risultati["u"])
    grafico_dinamico(risultati["dinamico"])
    grafico_metriche(risultati)
    print("\nFatto. 5 PNG in grafici/, dati cachati in dati/.")


if __name__ == "__main__":
    main()
