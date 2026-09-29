"""Reader for the LDM legacy-VTK particle dumps and the airborne/deposited/outflow masks.

File facts (report 1.1-2.4): legacy VTK 3.0, BINARY big-endian, DATASET POLYDATA, POINTS N float,
no cells, FIELD FieldData with 6 arrays in this order:
concentration(f32,1), p_type(i32,1), velocity(f32,3), e_turb(f32,1), k_turb(f32,1), concn(f32,1).
Deposited particles sit at z == 1e-4 with zero velocity and are never removed; particles that reach
the +x outflow are clamped near x ~ 1322.6 m (artifact, confirmed by the code owner) -> drop x >= 1315.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from srcloc_env import config

ARRAY_NAMES = ("concentration", "p_type", "velocity", "e_turb", "k_turb", "concn")


def deposited_mask(xyz: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Ground-frozen particles: z exactly DEPOSIT_Z and |velocity| == 0 (report 2.4)."""
    return (xyz[:, 2] == config.DEPOSIT_Z) & np.all(velocity == 0.0, axis=1)


def outflow_mask(xyz: np.ndarray) -> np.ndarray:
    """Outflow pile-up artifact beyond the LBM lattice end (x >= X_OUTFLOW)."""
    return xyz[:, 0] >= config.X_OUTFLOW


def airborne_mask(xyz: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Particles usable for airborne concentration: neither deposited nor in the outflow pile-up."""
    return ~deposited_mask(xyz, velocity) & ~outflow_mask(xyz)


@dataclass
class LdmFrame:
    """One LDM output file as numpy arrays (float32 / int32, N particles)."""

    index: int
    step: int
    xyz: np.ndarray            # (N, 3) m, fluid/LDM frame
    p_type: np.ndarray         # (N,) int32 source id 101..113
    velocity: np.ndarray       # (N, 3) turbulent fluctuation u' (NOT the mean wind), m/s
    concentration: np.ndarray  # (N,) same-source count within 7.5 m (integer valued)
    concn: np.ndarray          # (N,) same-source Wendland-C6 sum in lattice units
    k_turb: np.ndarray | None = None
    e_turb: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def deposited(self) -> np.ndarray:
        return deposited_mask(self.xyz, self.velocity)

    @property
    def outflow(self) -> np.ndarray:
        return outflow_mask(self.xyz)

    @property
    def airborne(self) -> np.ndarray:
        return airborne_mask(self.xyz, self.velocity)

    def select(self, mask: np.ndarray) -> "LdmFrame":
        """Return a new frame containing only the particles where mask is True."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v[mask] if isinstance(v, np.ndarray) else v
        return LdmFrame(**out)

    def counts_by_source(self, mask: np.ndarray | None = None) -> dict[int, int]:
        pt = self.p_type if mask is None else self.p_type[mask]
        ids, cnt = np.unique(pt, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, cnt)}

    def summary(self) -> dict:
        dep, out = self.deposited, self.outflow
        return {
            "index": self.index, "step": self.step, "n_total": self.n,
            "n_deposited": int(dep.sum()), "n_outflow": int(out.sum()),
            "n_airborne": int((~dep & ~out).sum()),
            "n_z_eq_deposit": int((self.xyz[:, 2] == config.DEPOSIT_Z).sum()),
            "n_x_gt_1322": int((self.xyz[:, 0] > 1322.0).sum()),
            "bbox_min": self.xyz.min(axis=0).tolist(), "bbox_max": self.xyz.max(axis=0).tolist(),
            "counts_by_source": self.counts_by_source(),
        }


def read_ldm_vtk(path: Path) -> dict[str, np.ndarray]:
    """Read one LDM file with VTK and return {'xyz': (N,3), <array name>: ...} as numpy arrays."""
    import vtk  # imported lazily: the NPZ path does not need VTK
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    pd = reader.GetOutput()
    if pd.GetNumberOfPoints() == 0:
        raise IOError(f"no points read from {path}")
    out = {"xyz": np.ascontiguousarray(vtk_to_numpy(pd.GetPoints().GetData()), dtype=np.float32)}
    point_data = pd.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        arr = point_data.GetArray(i)
        out[arr.GetName()] = np.ascontiguousarray(vtk_to_numpy(arr))
    missing = [n for n in ARRAY_NAMES if n not in out]
    if missing:
        raise IOError(f"{path}: missing arrays {missing}; found {sorted(out)}")
    return out


def frame_from_arrays(index: int, arrays: dict[str, np.ndarray]) -> LdmFrame:
    return LdmFrame(
        index=index, step=config.index_to_step(index),
        xyz=arrays["xyz"].astype(np.float32, copy=False),
        p_type=arrays["p_type"].astype(np.int32, copy=False),
        velocity=arrays["velocity"].astype(np.float32, copy=False),
        concentration=arrays["concentration"].astype(np.float32, copy=False),
        concn=arrays["concn"].astype(np.float32, copy=False),
        k_turb=None if arrays.get("k_turb") is None else arrays["k_turb"].astype(np.float32, copy=False),
        e_turb=None if arrays.get("e_turb") is None else arrays["e_turb"].astype(np.float32, copy=False),
    )


def load_frame(index: int) -> LdmFrame:
    """Load LDM file by index (0 -> step 15025, 599 -> step 30000) from the raw data folder."""
    return frame_from_arrays(index, read_ldm_vtk(config.ldm_path(index)))


def load_frame_from_npz(path: Path = config.LDM_30000_NPZ, index: int = config.N_FILES - 1) -> LdmFrame:
    """Load the pre-converted 30000 frame (keys: positions, p_type, concn, concentration, velocity)."""
    with np.load(path) as z:
        arrays = {"xyz": z["positions"], "p_type": z["p_type"], "velocity": z["velocity"],
                  "concentration": z["concentration"], "concn": z["concn"],
                  "k_turb": z["k_turb"] if "k_turb" in z.files else None,
                  "e_turb": z["e_turb"] if "e_turb" in z.files else None}
    return frame_from_arrays(index, arrays)