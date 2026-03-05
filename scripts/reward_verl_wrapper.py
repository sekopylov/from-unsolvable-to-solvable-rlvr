from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_REWARD_UTILS_PATH = _THIS_DIR / "reward_utils.py"
_spec = importlib.util.spec_from_file_location("reward_utils_local", _REWARD_UTILS_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load reward utils from {_REWARD_UTILS_PATH}")
_reward_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward_utils)
compute_reward = _reward_utils.compute_reward


def _normalize_gt(ground_truth: Any):
    # VERL reward loop may pass numpy arrays/scalars.
    try:
        import numpy as np

        if isinstance(ground_truth, np.ndarray):
            if ground_truth.size == 0:
                return None
            return ground_truth.reshape(-1)[0]
        if isinstance(ground_truth, np.generic):
            return ground_truth.item()
    except Exception:
        pass

    if isinstance(ground_truth, (list, tuple)):
        if len(ground_truth) == 0:
            return None
        return ground_truth[0]

    if isinstance(ground_truth, bytes):
        try:
            return ground_truth.decode("utf-8", errors="ignore")
        except Exception:
            return str(ground_truth)

    if isinstance(ground_truth, dict):
        for k in ("answer", "ground_truth", "target", "label"):
            if k in ground_truth:
                return _normalize_gt(ground_truth[k])

    return ground_truth


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = _normalize_gt(ground_truth)
    if gt is None:
        return 0.0

    try:
        ok = compute_reward(gt, solution_str)
        return float(bool(ok))
    except Exception:
        return 0.0
