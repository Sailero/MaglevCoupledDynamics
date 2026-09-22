"""Stored trajectory arrays plus sidecar metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from maglev_sim.plant.config import Config
from maglev_sim.plant.state import STATE_UNITS


@dataclass
class Trajectory:
    t: NDArray[np.float64]
    state: NDArray[np.float64]
    state_names: list[str]
    z_car: NDArray[np.float64]
    z_magnet: NDArray[np.float64]
    v_car: NDArray[np.float64]
    v_magnet: NDArray[np.float64]
    a_car: NDArray[np.float64]
    a_magnet: NDArray[np.float64]
    gap_raw: NDArray[np.float64]
    beam_displacement_at_magnet: NDArray[np.float64]
    beam_velocity_at_magnet: NDArray[np.float64]
    current: NDArray[np.float64]
    current_command: NDArray[np.float64]
    current_raw: NDArray[np.float64]
    magnetic_force: NDArray[np.float64]
    q: NDArray[np.float64]
    p: NDArray[np.float64]
    irregularity: NDArray[np.float64]
    force_x: NDArray[np.float64]
    force_P: NDArray[np.float64]
    xi: NDArray[np.float64]
    speed: NDArray[np.float64]
    saturated: NDArray[np.int8]
    completed: bool
    t_end_actual: float
    termination_reason: str
    integrator_message: str
    config: Config
    excitation_meta: dict[str, Any]
    seed: int | None = None
    trajectory_id: str = ""
    retune_log: list[dict[str, Any]] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=lambda: dict(STATE_UNITS))
    coordinate: str = "downward_positive"

    def metadata(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "completed": bool(self.completed),
            "t_end_actual": float(self.t_end_actual),
            "termination_reason": self.termination_reason,
            "integrator_message": self.integrator_message,
            "n_samples": int(self.t.size),
            "dt": float(self.t[1] - self.t[0]) if self.t.size > 1 else None,
            "state_names": self.state_names,
            "units": self.units,
            "coordinate": self.coordinate,
            "seed": self.seed,
            "model_version": self.config.meta.model_version,
            "parameter_source": self.config.meta.parameter_source,
            "notes": self.config.meta.notes,
            "config": self.config.to_dict(),
            "excitation": self.excitation_meta,
            "retune": self.retune_log,
            "initial_state": self.state[0].tolist() if self.state.size else [],
            "saturated_any": bool(np.any(self.saturated)),
        }

    @staticmethod
    def _json_default(obj: Any):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def save(self, directory: str | Path, trajectory_id: str | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tid = trajectory_id or self.trajectory_id or "trajectory"
        self.trajectory_id = tid
        npz_path = directory / f"{tid}.npz"
        json_path = directory / f"{tid}.json"
        np.savez_compressed(
            npz_path,
            t=self.t,
            state=self.state,
            z_car=self.z_car,
            z_magnet=self.z_magnet,
            v_car=self.v_car,
            v_magnet=self.v_magnet,
            a_car=self.a_car,
            a_magnet=self.a_magnet,
            gap_raw=self.gap_raw,
            beam_displacement_at_magnet=self.beam_displacement_at_magnet,
            beam_velocity_at_magnet=self.beam_velocity_at_magnet,
            current=self.current,
            current_command=self.current_command,
            current_raw=self.current_raw,
            magnetic_force=self.magnetic_force,
            q=self.q,
            p=self.p,
            irregularity=self.irregularity,
            force_x=self.force_x,
            force_P=self.force_P,
            xi=self.xi,
            speed=self.speed,
            saturated=self.saturated,
            state_names=np.asarray(self.state_names),
        )
        json_path.write_text(
            json.dumps(self.metadata(), indent=2, ensure_ascii=False, default=self._json_default),
            encoding="utf-8",
        )
        return npz_path

    @classmethod
    def load(cls, directory: str | Path, trajectory_id: str) -> Trajectory:
        directory = Path(directory)
        npz = np.load(directory / f"{trajectory_id}.npz")
        meta = json.loads((directory / f"{trajectory_id}.json").read_text(encoding="utf-8"))
        cfg = Config.from_dict(meta["config"])
        return cls(
            t=npz["t"],
            state=npz["state"],
            state_names=meta["state_names"],
            z_car=npz["z_car"],
            z_magnet=npz["z_magnet"],
            v_car=npz["v_car"],
            v_magnet=npz["v_magnet"],
            a_car=npz["a_car"],
            a_magnet=npz["a_magnet"],
            gap_raw=npz["gap_raw"],
            beam_displacement_at_magnet=npz["beam_displacement_at_magnet"],
            beam_velocity_at_magnet=npz["beam_velocity_at_magnet"],
            current=npz["current"],
            current_command=npz["current_command"],
            current_raw=npz["current_raw"],
            magnetic_force=npz["magnetic_force"],
            q=npz["q"],
            p=npz["p"],
            irregularity=npz["irregularity"],
            force_x=npz["force_x"],
            force_P=npz["force_P"],
            xi=npz["xi"],
            speed=npz["speed"],
            saturated=npz["saturated"],
            completed=bool(meta["completed"]),
            t_end_actual=float(meta["t_end_actual"]),
            termination_reason=str(meta["termination_reason"]),
            integrator_message=str(meta["integrator_message"]),
            config=cfg,
            excitation_meta=meta.get("excitation", {}),
            seed=meta.get("seed"),
            trajectory_id=trajectory_id,
            retune_log=meta.get("retune", []),
        )
