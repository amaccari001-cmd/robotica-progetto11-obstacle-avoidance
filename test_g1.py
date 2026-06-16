#!/usr/bin/env python3
"""Caricamento del robot Unitree G1 in MuJoCo: posa eretta e viewer.

Uso:
    python3 test_g1.py            # apre il viewer interattivo
    python3 test_g1.py --check    # verifica headless (no GUI)
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"


def load_model():
    """Carica il modello e posiziona il robot nella posa 'stand'."""
    if not MODEL_PATH.exists():
        sys.exit(f"ERRORE: modello non trovato in {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    try:
        key_id = model.keyframe("stand").id
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except KeyError:
        print("Attenzione: keyframe 'stand' non trovato, uso la posa di default.")
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)
    return model, data


def check():
    """Verifica headless: carica il modello e fa 100 passi di fisica."""
    model, data = load_model()
    print("OK - modello caricato correttamente")
    print(f"  file        : {MODEL_PATH}")
    print(f"  corpi (body): {model.nbody}")
    print(f"  giunti (DoF): {model.nv}")
    print(f"  attuatori   : {model.nu}")
    print(f"  timestep    : {model.opt.timestep * 1000:.2f} ms")

    for _ in range(100):
        mujoco.mj_step(model, data)

    altezza_bacino = data.qpos[2]
    print(f"  dopo 100 passi di simulazione: altezza base = {altezza_bacino:.3f} m")
    print("OK - la simulazione gira senza errori")


def view():
    """Apre il viewer interattivo di MuJoCo e simula in tempo reale."""
    model, data = load_model()
    print(f"Modello caricato: {MODEL_PATH}")
    print("Apertura del viewer MuJoCo... (chiudi la finestra per terminare)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    print("Viewer chiuso.")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        view()
