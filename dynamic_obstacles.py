#!/usr/bin/env python3
"""PF e DWA in ambiente con ostacoli MOBILI.

Uso:
    python3 dynamic_obstacles.py           # viewer, DWA in scenario dinamico
    python3 dynamic_obstacles.py pf        # viewer, Potential Fields
    python3 dynamic_obstacles.py dwa       # viewer, DWA
    python3 dynamic_obstacles.py --check   # headless, verifica rapida
    python3 dynamic_obstacles.py --compare # tabella: statico vs dinamico
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
import dwa

# --------------------------------------------------------------------------
# Scenario condiviso (identico a potential_fields.py / dwa.py)
# --------------------------------------------------------------------------
START = pf.START
GOAL = pf.GOAL
DT = pf.DT
V_MAX = pf.V_MAX
GOAL_TOL = pf.GOAL_TOL
MAX_STEPS = pf.MAX_STEPS
STUCK_STEPS = pf.STUCK_STEPS
OBST_HEIGHT = pf.OBST_HEIGHT
ROBOT_RADIUS = dwa.ROBOT_RADIUS

STATIC_OBS = list(pf.OBSTACLES)

# --------------------------------------------------------------------------
# Parametri dei 2 ostacoli dinamici
# --------------------------------------------------------------------------
OBS1_X = 1.8
OBS1_AMP = 1.05
OBS1_PERIOD = 6.0
OBS1_RADIUS = 0.37

OBS2_CX, OBS2_CY = 3.6, 0.0
OBS2_ORBIT = 0.85
OBS2_PERIOD = 6.0
OBS2_PHASE = math.pi / 2.0
OBS2_RADIUS = 0.34

N_PHASES = 8


def static_obstacles(t):
    """Scenario statico: ritorna i 3 ostacoli fissi (ignora t)."""
    return STATIC_OBS


def dynamic_obstacles(t, phase=0.0):
    """Scenario dinamico: ritorna le posizioni dei 2 ostacoli mobili a tempo t."""
    tt = t + phase
    y1 = OBS1_AMP * math.sin(2.0 * math.pi * tt / OBS1_PERIOD)
    ang = 2.0 * math.pi * tt / OBS2_PERIOD + OBS2_PHASE
    x2 = OBS2_CX + OBS2_ORBIT * math.cos(ang)
    y2 = OBS2_CY + OBS2_ORBIT * math.sin(ang)
    return [(OBS1_X, y1, OBS1_RADIUS), (x2, y2, OBS2_RADIUS)]


def dynamic_at(phase):
    """Closure: ritorna una funzione t -> ostacoli con la fase iniziale data."""
    return lambda t: dynamic_obstacles(t, phase)


def clearance(pos, obs):
    """Distanza minima dal robot (punto) alla superficie degli ostacoli."""
    return min(math.hypot(pos[0] - ox, pos[1] - oy) - r for (ox, oy, r) in obs)


# --------------------------------------------------------------------------
# Modello MuJoCo con i 2 ostacoli mobili come corpi 'mocap'
# --------------------------------------------------------------------------
def build_dynamic_model():
    """Scena G1 + 2 ostacoli mobili (mocap) + marcatore del goal."""
    spec = mujoco.MjSpec.from_file(str(pf.SCENE_PATH))
    world = spec.worldbody

    colori = [[0.95, 0.55, 0.10, 1.0],
              [0.65, 0.15, 0.80, 1.0]]
    for i, (ox, oy, r) in enumerate(dynamic_obstacles(0.0)):
        b = world.add_body()
        b.name = f"obs_mobile_{i}"
        b.mocap = True
        b.pos = [ox, oy, OBST_HEIGHT / 2.0]
        g = b.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [r, OBST_HEIGHT / 2.0, 0.0]
        g.rgba = colori[i]

    gm = world.add_geom()
    gm.name = "goal_marker"
    gm.type = mujoco.mjtGeom.mjGEOM_SPHERE
    gm.pos = [GOAL[0], GOAL[1], 0.15]
    gm.size = [0.15, 0.0, 0.0]
    gm.rgba = [0.10, 0.85, 0.20, 0.6]
    gm.contype = 0
    gm.conaffinity = 0

    return spec.compile()


def _open_viewer():
    """Costruisce il modello dinamico e apre il viewer (ritorna model, data, viewer, mocap_ids)."""
    model = build_dynamic_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("stand").id)
    mocap_ids = [model.body(f"obs_mobile_{i}").mocapid[0] for i in range(2)]
    viewer = mujoco.viewer.launch_passive(model, data)
    return model, data, viewer, mocap_ids


def _render(model, data, viewer, mocap_ids, obs, pos, yaw):
    """Aggiorna mocap (ostacoli) e qpos (robot) e disegna il frame."""
    for i, (ox, oy, _) in enumerate(obs):
        data.mocap_pos[mocap_ids[i]] = [ox, oy, OBST_HEIGHT / 2.0]
    pf.set_base(data, pos, yaw)
    mujoco.mj_forward(model, data)
    viewer.sync()


def _summary(esito, algoritmo, scenario, step, dist_goal, collision_steps):
    """Stampa il riepilogo finale di una simulazione."""
    print("-" * 104)
    testa = f"[{algoritmo} / scenario {scenario}]"
    if esito == "goal_raggiunto":
        print(f"{testa} GOAL RAGGIUNTO in {step} step  -  "
              f"distanza finale = {dist_goal:.3f} m  -  "
              f"step in collisione = {collision_steps}")
    elif esito in ("minimo_locale", "stallo"):
        print(f"{testa} {esito.upper()}: robot bloccato dopo {step} step "
              f"(dist_goal = {dist_goal:.3f} m)")
    elif esito == "max_steps":
        print(f"{testa} STOP: raggiunto MAX_STEPS senza arrivare al goal.")
    else:
        print(f"{testa} simulazione interrotta dopo {step} step.")


def _close_viewer(viewer):
    """Tiene aperta la finestra per ispezione, poi chiude in modo pulito."""
    print("(la finestra resta aperta - chiudila per terminare)")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
    viewer.close()


# --------------------------------------------------------------------------
# Simulazione: Potential Fields (robot olonomico)
# --------------------------------------------------------------------------
def run_pf(obstacle_fn, scenario, headless=True):
    """Esegue PF con ostacoli definiti da obstacle_fn(t)."""
    pos = START.astype(float).copy()
    yaw = 0.0

    model = data = viewer = None
    mocap_ids = []
    if not headless:
        model, data, viewer, mocap_ids = _open_viewer()

    dist_goal = float(np.linalg.norm(GOAL - pos))
    step, stuck = 0, 0
    esito = "interrotta"
    path_length = 0.0
    min_clearance = clearance(pos, obstacle_fn(0.0))
    collision_steps = 0

    print(f"[Potential Fields / scenario {scenario}]  "
          f"Start=({START[0]:.1f},{START[1]:.1f})  Goal=({GOAL[0]:.1f},{GOAL[1]:.1f})")
    print("-" * 104)

    while step < MAX_STEPS:
        if viewer is not None and not viewer.is_running():
            break

        obs = obstacle_fn(step * DT)
        pf.OBSTACLES = obs

        f_att = pf.attractive_force(pos)
        f_rep = pf.repulsive_force(pos)
        f_tot = f_att + f_rep
        speed = np.linalg.norm(f_tot)
        vel = f_tot / speed * V_MAX if speed > V_MAX else f_tot

        prev = pos
        pos = pos + vel * DT
        dist_goal = float(np.linalg.norm(GOAL - pos))
        path_length += float(np.linalg.norm(pos - prev))

        clr = clearance(pos, obs)
        min_clearance = min(min_clearance, clr)
        if clr < ROBOT_RADIUS:
            collision_steps += 1

        if np.linalg.norm(vel) > 1e-3:
            yaw = math.atan2(vel[1], vel[0])
        if viewer is not None:
            _render(model, data, viewer, mocap_ids, obs, pos, yaw)

        print(f"step {step:4d} | pos=({pos[0]:6.3f},{pos[1]:6.3f}) | "
              f"dist_goal={dist_goal:6.3f} | clearance={clr:6.3f} | "
              f"speed={np.linalg.norm(vel):5.3f}")

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

    _summary(esito, "Potential Fields", scenario, step, dist_goal, collision_steps)
    if viewer is not None:
        _close_viewer(viewer)

    return {
        "algoritmo": "Potential Fields", "scenario": scenario, "esito": esito,
        "step": step, "tempo_s": step * DT, "dist_finale": dist_goal,
        "lunghezza_path": path_length, "clearance_min": min_clearance,
        "collision_steps": collision_steps,
    }


# --------------------------------------------------------------------------
# Simulazione: DWA (robot a uniciclo)
# --------------------------------------------------------------------------
def run_dwa(obstacle_fn, scenario, headless=True):
    """Esegue DWA con ostacoli definiti da obstacle_fn(t)."""
    state = np.array([START[0], START[1], 0.0])
    v, omega = 0.0, 0.0

    model = data = viewer = None
    mocap_ids = []
    if not headless:
        model, data, viewer, mocap_ids = _open_viewer()

    dist_goal = float(np.linalg.norm(GOAL - state[:2]))
    step, stuck = 0, 0
    esito = "interrotta"
    path_length = 0.0
    min_clearance = clearance(state[:2], obstacle_fn(0.0))
    collision_steps = 0

    print(f"[DWA / scenario {scenario}]  "
          f"Start=({START[0]:.1f},{START[1]:.1f})  Goal=({GOAL[0]:.1f},{GOAL[1]:.1f})")
    print("-" * 104)

    while step < MAX_STEPS:
        if viewer is not None and not viewer.is_running():
            break

        obs = obstacle_fn(step * DT)
        dwa.OBSTACLES = obs

        v, omega, info = dwa.plan(state, v, omega)

        prev_xy = state[:2].copy()
        state = dwa.motion(state, v, omega, DT)
        state[2] = math.atan2(math.sin(state[2]), math.cos(state[2]))
        dist_goal = float(np.linalg.norm(GOAL - state[:2]))
        path_length += float(np.linalg.norm(state[:2] - prev_xy))

        clr = clearance(state[:2], obs)
        min_clearance = min(min_clearance, clr)
        if clr < ROBOT_RADIUS:
            collision_steps += 1

        if viewer is not None:
            _render(model, data, viewer, mocap_ids, obs, state[:2], state[2])

        print(f"step {step:4d} | pos=({state[0]:6.3f},{state[1]:6.3f}) | "
              f"dist_goal={dist_goal:6.3f} | clearance={clr:6.3f} | "
              f"v={v:5.3f} omega={omega:6.3f}")

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

    _summary(esito, "DWA", scenario, step, dist_goal, collision_steps)
    if viewer is not None:
        _close_viewer(viewer)

    return {
        "algoritmo": "DWA", "scenario": scenario, "esito": esito,
        "step": step, "tempo_s": step * DT, "dist_finale": dist_goal,
        "lunghezza_path": path_length, "clearance_min": min_clearance,
        "collision_steps": collision_steps,
    }


# --------------------------------------------------------------------------
# Confronto: statico vs dinamico, per entrambi gli algoritmi
# --------------------------------------------------------------------------
def _silent(funzione, *args):
    """Esegue una simulazione headless silenziando le stampe per-step."""
    with contextlib.redirect_stdout(io.StringIO()):
        return funzione(*args, headless=True)


def _aggrega(runs):
    """Aggrega le metriche di piu' run (le diverse fasi dello scenario dinamico)."""
    ok = [m for m in runs if m["esito"] == "goal_raggiunto"]
    base = ok if ok else runs

    def media(key, campione):
        return sum(m[key] for m in campione) / len(campione)

    return {
        "n": len(runs),
        "n_ok": len(ok),
        "step": media("step", base),
        "tempo_s": media("tempo_s", base),
        "dist_finale": media("dist_finale", runs),
        "lunghezza_path": media("lunghezza_path", base),
        "clearance_min": min(m["clearance_min"] for m in runs),
        "collision_steps": media("collision_steps", runs),
    }


def _col_statico(m):
    """Stringhe pre-formattate per una colonna 'statico' (run singolo)."""
    return {
        "esito": m["esito"],
        "step": str(m["step"]),
        "tempo_s": f"{m['tempo_s']:.2f}",
        "dist_finale": f"{m['dist_finale']:.3f}",
        "lunghezza_path": f"{m['lunghezza_path']:.3f}",
        "clearance_min": f"{m['clearance_min']:.3f}",
        "collision_steps": str(m["collision_steps"]),
    }


def _col_dinamico(agg):
    """Stringhe pre-formattate per una colonna 'dinamico' (media su N fasi)."""
    return {
        "esito": f"{agg['n_ok']}/{agg['n']} al goal",
        "step": f"{agg['step']:.0f}",
        "tempo_s": f"{agg['tempo_s']:.2f}",
        "dist_finale": f"{agg['dist_finale']:.3f}",
        "lunghezza_path": f"{agg['lunghezza_path']:.3f}",
        "clearance_min": f"{agg['clearance_min']:.3f}",
        "collision_steps": f"{agg['collision_steps']:.0f}",
    }


def compare():
    """Confronto PF vs DWA, scenario statico vs dinamico (mediato su N_PHASES)."""
    print("=" * 86)
    print("CONFRONTO  -  scenario STATICO vs DINAMICO  -  Potential Fields vs DWA")
    print("=" * 86)
    print(f"Start = ({START[0]:.1f}, {START[1]:.1f})   "
          f"Goal = ({GOAL[0]:.1f}, {GOAL[1]:.1f})")
    print("Scenario statico : 3 ostacoli fissi")
    print("Scenario dinamico: ostacolo 1 lineare su Y (avanti/indietro), "
          "ostacolo 2 circolare")
    print(f"                   media su {N_PHASES} fasi iniziali degli ostacoli")
    print()

    print("  esecuzione simulazioni headless...", end=" ", flush=True)
    m_pf_s = _silent(run_pf, static_obstacles, "statico")
    m_dwa_s = _silent(run_dwa, static_obstacles, "statico")
    fasi = [i * OBS1_PERIOD / N_PHASES for i in range(N_PHASES)]
    runs_pf_d = [_silent(run_pf, dynamic_at(ph), "dinamico") for ph in fasi]
    runs_dwa_d = [_silent(run_dwa, dynamic_at(ph), "dinamico") for ph in fasi]
    agg_pf_d = _aggrega(runs_pf_d)
    agg_dwa_d = _aggrega(runs_dwa_d)
    print("fatto")
    print()

    c_pf_s = _col_statico(m_pf_s)
    c_pf_d = _col_dinamico(agg_pf_d)
    c_dwa_s = _col_statico(m_dwa_s)
    c_dwa_d = _col_dinamico(agg_dwa_d)

    righe = [
        ("Esito",                  "esito"),
        ("Step al goal",           "step"),
        ("Tempo simulato [s]",     "tempo_s"),
        ("Distanza finale [m]",    "dist_finale"),
        ("Lunghezza percorso [m]", "lunghezza_path"),
        ("Clearance minima [m]",   "clearance_min"),
        ("Step in collisione",     "collision_steps"),
    ]
    print(f"{'':<26}{'POTENTIAL FIELDS':^30}{'DWA':^30}")
    print(f"{'METRICA':<26}{'statico':>15}{'dinamico':>15}"
          f"{'statico':>15}{'dinamico':>15}")
    print("-" * 86)
    for label, key in righe:
        print(f"{label:<26}{c_pf_s[key]:>15}{c_pf_d[key]:>15}"
              f"{c_dwa_s[key]:>15}{c_dwa_d[key]:>15}")
    print("-" * 86)
    print("Colonne 'dinamico': valori medi su N fasi; clearance = caso peggiore.")
    print()
    _verdetto(agg_pf_d, agg_dwa_d)


def _verdetto(agg_pf, agg_dwa):
    """Commento testuale: quale algoritmo regge meglio con ostacoli mobili."""
    print("VERDETTO  -  chi regge meglio con ostacoli in movimento?")
    print(f"  Potential Fields : {agg_pf['n_ok']}/{agg_pf['n']} al goal  |  "
          f"collisioni medie = {agg_pf['collision_steps']:.0f} step  |  "
          f"clearance peggiore = {agg_pf['clearance_min']:.3f} m")
    print(f"  DWA              : {agg_dwa['n_ok']}/{agg_dwa['n']} al goal  |  "
          f"collisioni medie = {agg_dwa['collision_steps']:.0f} step  |  "
          f"clearance peggiore = {agg_dwa['clearance_min']:.3f} m")

    if agg_pf["n_ok"] != agg_dwa["n_ok"]:
        vincitore = "Potential Fields" if agg_pf["n_ok"] > agg_dwa["n_ok"] else "DWA"
        motivo = "raggiunge il goal piu' spesso"
    elif abs(agg_pf["collision_steps"] - agg_dwa["collision_steps"]) > 1e-6:
        if agg_pf["collision_steps"] < agg_dwa["collision_steps"]:
            vincitore, motivo = "Potential Fields", "collide molto meno"
        else:
            vincitore, motivo = "DWA", "collide molto meno"
    else:
        vincitore, motivo = None, ""

    if vincitore is None:
        print("  => I due algoritmi reggono in modo equivalente.")
    else:
        print(f"  => Regge meglio: {vincitore} ({motivo}).")
    print("=" * 86)


# --------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]

    if "--compare" in args:
        compare()
    elif "--check" in args:
        run_dwa(dynamic_obstacles, "dinamico", headless=True)
    elif "pf" in args:
        run_pf(dynamic_obstacles, "dinamico", headless=False)
    else:
        run_dwa(dynamic_obstacles, "dinamico", headless=False)


if __name__ == "__main__":
    main()
