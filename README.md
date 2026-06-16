# Obstacle Avoidance Reattivo senza Mappa — Unitree G1

Progetto del corso di Robotica (A.A. 2025/2026), Università degli Studi di Brescia.
Autori: Andrea Maccari, Alice Finotto.

Questo progetto mette a confronto due algoritmi di navigazione reattiva — **Potential Fields (PF)** e **Dynamic Window Approach (DWA)** — sul robot umanoide Unitree G1 simulato in MuJoCo. Il robot deve raggiungere un punto di arrivo noto evitando gli ostacoli, usando solo i sensori locali: non costruisce nessuna mappa.

Il confronto avviene su quattro scenari: slalom, corridoio stretto, ostacolo a U e ostacoli in movimento.

## Cosa serve prima di iniziare

Serve **Python 3** installato. Per verificarlo:

```bash
python3 --version
```

Poi si installano le tre librerie necessarie:

```bash
pip install mujoco numpy matplotlib
```

Infine serve il modello del robot Unitree G1. Deve trovarsi in una cartella chiamata `unitree_g1/` dentro al progetto, e deve contenere il file `unitree_g1/scene.xml`.

Se la cartella `unitree_g1/` è già presente nel progetto, non bisogna fare nulla. Se manca, si scarica così:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
cp -r mujoco_menagerie/unitree_g1 ./unitree_g1
```

## Come avviare il programma

Tutti i comandi vanno dati dalla cartella del progetto. Per spostarsi dentro la cartella:

```bash
cd robotica-progetto11-obstacle-avoidance
```

### Passo 1 — Controllare che il robot funzioni

Prima di tutto conviene verificare che MuJoCo e il modello del robot siano a posto. Questo comando apre una finestra in cui il robot G1 deve comparire in piedi:

```bash
python3 test_g1.py
```

Se la finestra mostra il robot fermo in piedi, è tutto pronto. Se invece dà errore, di solito manca la cartella `unitree_g1/` (vedi sopra) o una delle librerie.

### Passo 2 — Scenario slalom

Il robot attraversa tre ostacoli a zig-zag fino al punto di arrivo.

```bash
python3 potential_fields.py     # usa l'algoritmo Potential Fields
python3 dwa.py                  # usa l'algoritmo DWA
python3 dwa.py --compare        # esegue entrambi e stampa una tabella di confronto
```

### Passo 3 — Corridoio stretto e ostacolo a U

```bash
python3 scenari_statici.py corridoio pf     # corridoio, Potential Fields
python3 scenari_statici.py corridoio dwa    # corridoio, DWA
python3 scenari_statici.py u pf             # ostacolo a U, Potential Fields
python3 scenari_statici.py u dwa            # ostacolo a U, DWA
python3 scenari_statici.py --compare        # tabella di confronto su entrambi
```

### Passo 4 — Ostacoli in movimento

```bash
python3 dynamic_obstacles.py pf         # Potential Fields
python3 dynamic_obstacles.py dwa        # DWA
python3 dynamic_obstacles.py --compare  # tabella di confronto
```

### Passo 5 — Generare i grafici

Questo comando esegue le simulazioni e salva i grafici di confronto in formato PNG nella cartella `grafici/`:

```bash
python3 genera_grafici.py
```

La prima volta calcola tutto da capo (può richiedere qualche minuto). Le volte successive riusa i dati già calcolati. Per forzare un nuovo calcolo da zero:

```bash
python3 genera_grafici.py --rigenera
```

## Note

Le finestre di simulazione hanno bisogno di un ambiente grafico. Se si lavora su un server senza schermo, usare le versioni `--compare`, che stampano i risultati a testo senza aprire la finestra.
