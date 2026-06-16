#!/usr/bin/env python3
"""Navigazione reattiva del robot Unitree G1 con i Potential Fields.

Uso:
    python3 potential_fields.py            # viewer interattivo
    python3 potential_fields.py --check    # headless (no GUI)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# --------------------------------------------------------------------------
# Configurazione dell'ambiente
# --------------------------------------------------------------------------
SCENE_PATH = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"

START = np.array([0.0, 0.0])
GOAL = np.array([5.0, 0.0])

OBSTACLES = [
    (1.5,  0.60, 0.30),
    (2.9, -1.05, 0.32),
    (4.0,  0.60, 0.30),
]

# --------------------------------------------------------------------------
# Parametri dell'algoritmo Potential Fields
# --------------------------------------------------------------------------
K_ATT = 1.5
K_REP = 0.5
RHO_0 = 0.90
F_REP_MAX = 3.0
V_MAX = 0.70
DT = 0.02
GOAL_TOL = 0.12
MAX_STEPS = 4000
STUCK_STEPS = 200
OBST_HEIGHT = 1.2
BASE_HEIGHT = 0.793


# --------------------------------------------------------------------------
# Costruzione del modello: scena G1 + ostacoli + marcatore del goal
# --------------------------------------------------------------------------
def build_model():
    """Carica scene.xml e vi inietta ostacoli e goal tramite MjSpec."""
    if not SCENE_PATH.exists():
        sys.exit(f"ERRORE: scena non trovata in {SCENE_PATH}")

    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    world = spec.worldbody

    for i, (ox, oy, r) in enumerate(OBSTACLES):
        g = world.add_geom()
        g.name = f"ostacolo_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.pos = [ox, oy, OBST_HEIGHT / 2.0]
        g.size = [r, OBST_HEIGHT / 2.0, 0.0]
        g.rgba = [0.85, 0.20, 0.15, 1.0]

    gm = world.add_geom()
    gm.name = "goal_marker"
    gm.type = mujoco.mjtGeom.mjGEOM_SPHERE
    gm.pos = [GOAL[0], GOAL[1], 0.15]
    gm.size = [0.15, 0.0, 0.0]
    gm.rgba = [0.10, 0.85, 0.20, 0.6]
    gm.contype = 0
    gm.conaffinity = 0

    return spec.compile()


def set_base(data, pos, yaw):
    """Posiziona la base in (x, y) con angolo di imbardata yaw."""
    data.qpos[0] = pos[0]
    data.qpos[1] = pos[1]
    data.qpos[2] = BASE_HEIGHT
    data.qpos[3] = math.cos(yaw / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = math.sin(yaw / 2.0)


# --------------------------------------------------------------------------
# Campi di potenziale
# --------------------------------------------------------------------------
def attractive_force(pos):
    """Forza attrattiva verso il goal, proporzionale alla distanza, saturata a V_MAX."""
    f = K_ATT * (GOAL - pos)
    n = np.linalg.norm(f)
    if n > V_MAX:
        f = f / n * V_MAX
    return f


def repulsive_force(pos):
    """Somma delle forze repulsive (Khatib) generate dagli ostacoli entro RHO_0."""
    f_tot = np.zeros(2)
    for (ox, oy, r) in OBSTACLES:
        to_robot = pos - np.array([ox, oy])
        dist_center = np.linalg.norm(to_robot)
        d = dist_center - r
        if d < RHO_0:
            d_safe = max(d, 0.05)
            mag = K_REP * (1.0 / d_safe - 1.0 / RHO_0) / (d_safe ** 2)
            mag = min(mag, F_REP_MAX)
            direction = to_robot / max(dist_center, 1e-9)
            f_tot += mag * direction
    return f_tot


def clearance_to_obstacles(pos):
    """Distanza minima dal robot (punto) alla superficie degli ostacoli."""
    return min(float(np.linalg.norm(pos - np.array([ox, oy]))) - r
               for (ox, oy, r) in OBSTACLES)


# --------------------------------------------------------------------------
# Ciclo di simulazione
# --------------------------------------------------------------------------
def run(headless=False):
    """Esegue una simulazione PF e ritorna il dict di metriche."""
    model = build_model()
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("stand").id)
    pos = START.astype(float).copy()
    yaw = 0.0
    set_base(data, pos, yaw)

    mujoco.mj_forward(model, data)

    print(f"Start = ({START[0]:.2f}, {START[1]:.2f})   "
          f"Goal = ({GOAL[0]:.2f}, {GOAL[1]:.2f})")
    for i, (ox, oy, r) in enumerate(OBSTACLES):
        print(f"  ostacolo_{i}: centro=({ox:.2f}, {oy:.2f})  raggio={r:.2f} m")
    print("-" * 104)

    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    dist_goal = float(np.linalg.norm(GOAL - pos))
    step = 0
    stuck = 0
    esito = "interrotta"
    path_length = 0.0
    min_clearance = clearance_to_obstacles(pos)

    while step < MAX_STEPS:
        if viewer is not None and not viewer.is_running():
            break

        f_att = attractive_force(pos)
        f_rep = repulsive_force(pos)
        f_tot = f_att + f_rep

        speed = np.linalg.norm(f_tot)
        vel = f_tot / speed * V_MAX if speed > V_MAX else f_tot

        prev_pos = pos
        pos = pos + vel * DT
        dist_goal = float(np.linalg.norm(GOAL - pos))
        path_length += float(np.linalg.norm(pos - prev_pos))
        min_clearance = min(min_clearance, clearance_to_obstacles(pos))

        if np.linalg.norm(vel) > 1e-3:
            yaw = math.atan2(vel[1], vel[0])

        set_base(data, pos, yaw)
        mujoco.mj_forward(model, data)
        if viewer is not None:
            viewer.sync()

        print(f"step {step:4d} | pos=({pos[0]:6.3f},{pos[1]:6.3f}) | "
              f"dist_goal={dist_goal:6.3f} | "
              f"F_att=({f_att[0]:6.3f},{f_att[1]:6.3f}) | "
              f"F_rep=({f_rep[0]:6.3f},{f_rep[1]:6.3f}) | "
              f"F_tot=({f_tot[0]:6.3f},{f_tot[1]:6.3f})")

        if dist_goal < GOAL_TOL:
            esito = "goal_raggiunto"
            break

        stuck = stuck + 1 if np.linalg.norm(vel) < 0.02 else 0
        if stuck >= STUCK_STEPS:
            esito = "minimo_locale"
            break

        step += 1
        if viewer is not None:
            time.sleep(DT)
    else:
        esito = "max_steps"

    print("-" * 104)
    if esito == "goal_raggiunto":
        print(f"GOAL RAGGIUNTO in {step} step  -  distanza finale = {dist_goal:.3f} m")
    elif esito == "minimo_locale":
        print(f"MINIMO LOCALE: robot bloccato dopo {step} step "
              f"(dist_goal = {dist_goal:.3f} m). "
              f"E' il limite noto dei Potential Fields.")
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
        "algoritmo": "Potential Fields",
        "esito": esito,
        "step": step,
        "tempo_s": step * DT,
        "dist_finale": dist_goal,
        "lunghezza_path": path_length,
        "clearance_min": min_clearance,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    run(headless="--check" in sys.argv)
