"""Визуализатор плотного варианта ``omega5``."""

from __future__ import annotations

from typing import Any, Mapping

from ..omega5.visualizer import OmegaVisualizer


class DenseOmegaVisualizer(OmegaVisualizer):
    experiment = "omega5_dense"

    def rule_info(self, values: Mapping[str, Any]) -> dict[str, Any]:
        info = super().rule_info(values)
        states = int(values.get("grid2d.states", 8))
        profile = {}
        for state in range(states):
            lo = values.get(f"dense.share{state}_lo")
            hi = values.get(f"dense.share{state}_hi")
            if lo is not None and hi is not None:
                profile[str(state)] = [float(lo), float(hi)]
        info["dense_profile"] = profile
        info["dense_profile_text"] = "\n".join(
            f"состояние {state}: {lo * 100:.1f}%..{hi * 100:.1f}%"
            for state, (lo, hi) in sorted(profile.items(), key=lambda item: int(item[0]))
        )
        info["dynamics"] = {
            "min_change": float(values.get("dense.min_change", 0.0)),
            "max_change": float(values.get("dense.max_change", 1.0)),
            "min_present_states": int(values.get("dense.min_present_states", 0)),
        }
        info["density"] = {
            "min": float(values.get("dense.density_min", 0.0)),
            "max": float(values.get("dense.density_max", 1.0)),
        }
        return info


__all__ = ["DenseOmegaVisualizer"]
