# LBM_MARL (srcloc_env)

Multi-drone radiological **source-term estimation (STE)** on a high-fidelity urban dispersion dataset
(LBM wind field + Lagrangian dispersion model, Leipzig building geometry).

Pipeline: point-measurement sensor (Wendland C6 gather + Poisson counts) → Rao-Blackwellised particle
filter over the source position with the release-rate/sensitivity product marginalised → GMM belief
summary → Gymnasium environment for single- and multi-drone policies (parameter-shared PPO).

Target: ICRS-15 presentation (submission 2026-10-19). Plan and data report live next to the data:
`F:\김도현박사님 자료\JH\ICRS15_연구수행계획.md`, `F:\김도현박사님 자료\JH\JH_데이터분석_및_연구환경설계.md`.

## Layout

```
srcloc_env/                package (flat layout)
  config.py                every path and constant, with its source annotated
  io/                      ldm_reader (legacy VTK -> arrays + airborne/outflow masks), stl_tools
  preprocess/              gridder (per-source concentration slabs)
  field/                   concentration_field (frame/scenario queries), wind (LBM wind lookup)
  sensor/                  detector (gather -> particles/m^3 -> Poisson counts)
  pf/                      forward_model, particle_filter (RB-PF), gmm_summary
  env/                     drone, source_env (Gymnasium), multi_agent
  baselines/               policies (random, lawnmower, greedy_map, gmm_infotaxis, oracle_loiter), coverage, planning
  eval/                    episodes (common seed lists), run_eval (batch runner), metrics
  rl/                      ppo (actor-critic, GAE, update), rollout (worker pool), train (CLI), ckpt_eval, ppo_policy
  scripts/                 CLI entry points: convert_ldm, validate_*, gate_report, fig_*
tests/                     pytest unit tests on synthetic data (no raw data needed)
docs/                      references.md (which paper's method is used where and why), notes
```

## Setup (Windows, PowerShell)

```powershell
Set-Location "F:\김도현박사님 자료\JH\srcloc_env"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; GPU build optional
$env:PYTHONIOENCODING = 'utf-8'
pytest
```

Data-dependent validation scripts (raw data folder is configured in `srcloc_env/config.py`):

```powershell
python -m srcloc_env.scripts.validate_c6 --n 2000 --seed 0                        # T0-1: reproduce the file's concn
python -m srcloc_env.scripts.convert_ldm --index 400 599 --slab-z 15 12.5 17.5    # cache + slabs
```

## Conventions

- Coordinates: fluid/LDM frame (x 0–1315 m, y ±657.5 m, z from ground 0). The STL is translated by
  (+689.26, −11.48, 0) m into this frame.
- "index step" = the number in `LDM_{step}stp.vtk`; 1 LDM step = 0.25 s (confirmed by the code owner);
  whether 1 LDM step = 10 index steps (interpretation A, default) is still being confirmed.
- Airborne particles exclude deposited ones (z == 1e-4 and zero velocity) and the outflow pile-up (x >= 1315).

## Training and evaluation (PPO)

```powershell
python -m srcloc_env.rl.train --n-drones 1 --run-seed 1 --run-name m1_s1 --total-steps 1000000      # M1
python -m srcloc_env.rl.train --n-drones 2 --run-seed 1 --run-name m2_s1 --total-steps 500000 `
       --init-from <cache>/train/m1_s1/final.pt                                                    # M2 from the M1 weights
python -m srcloc_env.eval.run_eval --methods gmm_infotaxis --ppo ppo_m1=<cache>/train/m1_s1/final.pt --n-drones 1 --tag final
```

Runs write to `<cache>/train/<run-name>` (config.json, train_log.csv, episodes.csv, checkpoints, ckpt_eval), where `<cache>` is
`config.CACHE_DIR`. Settings and rationale: `docs/training_evaluation_spec.md` section 6.
