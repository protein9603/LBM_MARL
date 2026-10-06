"""Compatibility shim: the module was renamed to pf/sph_adjoint.py on 2026-10-06 (the flow solver is SPH, not LBM)."""
from srcloc_env.pf.sph_adjoint import *   # noqa: F401,F403
from srcloc_env.pf.sph_adjoint import AdjointParams, AdvectionDiffusionOperator, SphAdjointModel  # noqa: F401

LbmAdjointModel = SphAdjointModel
