#!/usr/bin/env python3
"""Navigazione reattiva del robot Unitree G1 con il Dynamic Window Approach.

Uso:
    python3 dwa.py            # viewer interattivo
    python3 dwa.py --check    # headless (no GUI)
    python3 dwa.py --compare  # confronto tabellare con Potential Fields
"""

import contextlib
import io
import math
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

import potential_fields as pf

# --------------------------------------------------------------------------
# Scenario condiviso con potential_fields.py
# --------------------------------------------------------------------------
START = pf.START
GOAL = pf.GOAL
OBSTACLES = pf.OBSTACLES
build_model = pf.build_model
set_base = pf.set_base
clearance_to_obstacles = pf.clearance_to_obstacles

DT = pf.DT
V_MAX = pf.V_MAX
GOAL_TOL = pf.GOAL_TOL
MAX_STEPS = pf.MAX_STEPS
STUCK_STEPS = pf.STUCK_STEPS

# --------------------------------------------------------------------------
# Parametri cinematici del robot (modello a uniciclo)
# --------------------------------------------------------------------------
V_MIN = 0.0
OMEGA_MAX = 2.0
A_V = 2.5
A_OMEGA = 6.0
ROBOT_RADIUS = 0.30

# --------------------------------------------------------------------------
# Parametri dell'algoritmo DWA
# --------------------------------------------------------------------------
N_V = 6
N_OMEGA = 15
PREDICT_TIME = 2.0
PREDICT_DT = 0.10

W_GOAL = 1.0
W_CLEAR = 0.8
W_SPEED = 0.2


# --------------------------------------------------------------------------
# Modello cinematico a uniciclo
# --------------------------------------------------------------------------
def motion(state, v, omega, dt):
    """Avanza lo stato (x, y, theta) di un passo dt con comandi (v, omega)."""
    x, y, theta = state
    return np.array([
        x + v * math.cos(theta) * dt,
        y + v * math.sin(theta) * dt,
        theta + omega * dt,
    ])


def body_clearance(x, y):
    """Gioco fra corpo del robot in (x, y) e l'ostacolo piu' vicino (negativo = collisione)."""
    return min(math.hypot(x - ox, y - oy) - r - ROBOT_RADIUS
               for (ox, oy, r) in OBSTACLES)


# --------------------------------------------------------------------------
# Nucleo del DWA
# --------------------------------------------------------------------------
def dynamic_window(v_cur, omega_cur):
    """Intervalli di v e omega raggiungibili in un passo DT."""
    v_lo = max(V_MIN, v_cur - A_V * DT)
    v_hi = min(V_MAX, v_cur + A_V * DT)
    w_lo = max(-OMEGA_MAX, omega_cur - A_OMEGA * DT)
    w_hi = min(OMEGA_MAX, omega_cur + A_OMEGA * DT)
    return v_lo, v_hi, w_lo, w_hi


def rollout(state, v, omega):
    """Simula la traiettoria a (v, omega) costanti per PREDICT_TIME e ritorna (dist_goal_min, clearance_min)."""
    s = np.asarray(state, dtype=float)
    min_clear = body_clearance(s[0], s[1])
    closest_goal = float(np.linalg.norm(GOAL - s[:2]))
    for _ in range(int(PREDICT_TIME / PREDICT_DT)):
        s = motion(s, v, omega, PREDICT_DT)
        min_clear = min(min_clear, body_clearance(s[0], s[1]))
        closest_goal = min(closest_goal, float(np.linalg.norm(GOAL - s[:2])))
    return closest_goal, min_clear


def plan(state, v_cur, omega_cur):
    """Esegue un passo di DWA e ritorna (v, omega, info)."""
    v_lo, v_hi, w_lo, w_hi = dynamic_window(v_cur, omega_cur)
    vs = np.linspace(v_lo, v_hi, N_V)
    ws = np.linspace(w_lo, w_hi, N_OMEGA)

    dist_now = float(np.linalg.norm(GOAL - state[:2]))

    valide = []
    for v in vs:
        for w in ws:
            closest_goal, min_clear = rollout(state, v, w)
            if min_clear < 0.0:
                continue
            avanzamento = dist_now - closest_goal
            valide.append((v, w, avanzamento, min_clear))

    if not valide:
        best, best_clear = (0.0, 0.0), -1e9
        for v in vs:
            for w in ws:
                _, mc = rollout(state, v, w)
                if mc > best_clear:
                    best_clear, best = mc, (v, w)
        return best[0], best[1], {"valide": 0, "score": 0.0}

    avanz = [c[2] for c in valide]
    clear = [c[3] for c in valide]
    a_lo, a_hi = min(avanz), max(avanz)
    c_lo, c_hi = min(clear), max(clear)

    def norm(x, lo, hi):
        return (x - lo) / (hi - lo) if hi > lo else 1.0

    best, best_score = valide[0], -1e9
    for cand in valide:
        v, w, avanzamento, cl = cand
        score = (W_GOAL * norm(avanzamento, a_lo, a_hi)
                 + W_CLEAR * norm(cl, c_lo, c_hi)
                 + W_SPEED * (v / V_MAX))
        if score > best_score:
            best_score, best = score, cand

    return best[0], best[1], {"valide": len(valide), "score": best_score}


# --------------------------------------------------------------------------
# Ciclo di simulazione
# --------------------------------------------------------------------------
def run(headless=False):
    """Esegue una simulazione DWA e ritorna il dict di metriche."""
    model = build_model()
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("stand").id)
    state = np.array([START[0], START[1], 0.0])
    v, omega = 0.0, 0.0
    set_base(data, state[:2], state[2])
    mujoco.mj_forward(model, data)

    print(f"Start = ({START[0]:.2f}, {START[1]:.2f})   "
          f"Goal = ({GOAL[0]:.2f}, {GOAL[1]:.2f})   [DWA]")
    for i, (ox, oy, r) in enumerate(OBSTACLES):
        print(f"  ostacolo_{i}: centro=({ox:.2f}, {oy:.2f})  raggio={r:.2f} m")
    print("-" * 104)

    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    dist_goal = float(np.linalg.norm(GOAL - state[:2]))
    step = 0
    stuck = 0
    esito = "interrotta"
    path_length = 0.0
    min_clearance = clearance_to_obstacles(state[:2])

    while step < MAX_STEPS:
        if viewer is not None and not viewer.is_running():
            break

        v, omega, info = plan(state, v, omega)

        prev_xy = state[:2].copy()
        state = motion(state, v, omega, DT)
        state[2] = math.atan2(math.sin(state[2]), math.cos(state[2]))

        dist_goal = float(np.linalg.norm(GOAL - state[:2]))
        path_length += float(np.linalg.norm(state[:2] - prev_xy))
        min_clearance = min(min_clearance, clearance_to_obstacles(state[:2]))

        set_base(data, state[:2], state[2])
        mujoco.mj_forward(model, data)
        if viewer is not None:
            viewer.sync()

        print(f"step {step:4d} | pos=({state[0]:6.3f},{state[1]:6.3f}) | "
              f"theta={state[2]:6.3f} | dist_goal={dist_goal:6.3f} | "
              f"v={v:5.3f} | omega={omega:6.3f}")

        if dist_goal < GOAL_TOL:
            esito = "goal_raggiunto"
            break

        stuck = stuck + 1 if v < 0.02 else 0
        if stuck >= STUCK_STEPS:
            esito = "stallo"
            break

        step += 1
        if viewer is not None:
            time.sleep(DT)
    else:
        esito = "max_steps"

    print("-" * 104)
    if esito == "goal_raggiunto":
        print(f"GOAL RAGGIUNTO in {step} step  -  distanza finale = {dist_goal:.3f} m")
    elif esito == "stallo":
        print(f"STALLO: robot fermo dopo {step} step "
              f"(dist_goal = {dist_goal:.3f} m). "
              f"Il DWA non trova traiettorie libere utili.")
    elif esito == "max_steps":
        print(f"STOP: raggiunto MAX_STEPS ({MAX_STEPS}) senza arrivare al goal.")
    else:
        print(f"Simulazione interrotta dall'utente dopo {step} step.")

    if viewer is not None:
        print("(la finestra resta aperta - chiudila per terminare il programma)")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()

    return {
        "algoritmo": "DWA",
        "esito": esito,
        "step": step,
        "tempo_s": step * DT,
        "dist_finale": dist_goal,
        "lunghezza_path": path_length,
        "clearance_min": min_clearance,
    }


# --------------------------------------------------------------------------
# Confronto diretto DWA vs Potential Fields
# --------------------------------------------------------------------------
def _run_silent(funzione):
    """Esegue una run() headless silenziando le stampe per-step."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return funzione(headless=True)


def compare():
    """Esegue PF e DWA sullo stesso scenario e stampa la tabella di confronto."""
    print("Confronto DWA vs Potential Fields sullo stesso scenario")
    print(f"Start = ({START[0]:.2f}, {START[1]:.2f})   "
          f"Goal = ({GOAL[0]:.2f}, {GOAL[1]:.2f})   "
          f"Ostacoli = {len(OBSTACLES)}")
    print()

    print("  esecuzione Potential Fields (headless)...", end=" ")
    m_pf = _run_silent(pf.run)
    print(m_pf["esito"])
    print("  esecuzione DWA (headless)...", end=" ")
    m_dwa = _run_silent(run)
    print(m_dwa["esito"])

    righe = [
        ("Esito",                  "esito",          None),
        ("Step al goal",           "step",           "d"),
        ("Tempo simulato [s]",     "tempo_s",        ".2f"),
        ("Distanza finale [m]",    "dist_finale",    ".3f"),
        ("Lunghezza percorso [m]", "lunghezza_path", ".3f"),
        ("Clearance minima [m]",   "clearance_min",  ".3f"),
    ]
    print()
    print(f"{'METRICA':<24} | {'Potential Fields':>18} | {'DWA':>18}")
    print("-" * 68)
    for label, key, fmt in righe:
        a, b = m_pf[key], m_dwa[key]
        if fmt is None:
            print(f"{label:<24} | {str(a):>18} | {str(b):>18}")
        else:
            print(f"{label:<24} | {a:>18{fmt}} | {b:>18{fmt}}")
    print("-" * 68)

    if m_pf["esito"] == "goal_raggiunto" and m_dwa["esito"] == "goal_raggiunto":
        piu_veloce = "DWA" if m_dwa["step"] < m_pf["step"] else "Potential Fields"
        piu_sicuro = ("DWA" if m_dwa["clearance_min"] > m_pf["clearance_min"]
                      else "Potential Fields")
        print(f"Entrambi raggiungono il goal. Piu' rapido: {piu_veloce}; "
              f"clearance maggiore: {piu_sicuro}.")
    else:
        print("Nota: almeno un algoritmo non ha raggiunto il goal "
              "(vedi colonna 'Esito').")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    if "--compare" in sys.argv:
        compare()
    else:
        run(headless="--check" in sys.argv)
