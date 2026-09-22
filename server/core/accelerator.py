"""Accelerator detection — CUDA, Apple Silicon (MPS) or CPU.

Single source of truth for "what can we train and embed on". Everything else
reads devices from here instead of shelling out to ``nvidia-smi`` directly.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any, Dict, List, Optional

CUDA = "cuda"
MPS = "mps"
CPU = "cpu"

_SMI_TIMEOUT_S = 5
_MB = 1024 * 1024


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def backend() -> str:
    """Active backend. ``KONI_DEVICE`` overrides detection."""
    forced = (os.getenv("KONI_DEVICE") or "").strip().lower()
    if forced in (CUDA, MPS, CPU):
        return forced
    torch = _torch()
    if torch is None:
        return CUDA if shutil.which("nvidia-smi") else CPU
    if torch.cuda.is_available():
        return CUDA
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return MPS
    return CPU


def torch_device() -> str:
    """Device string for ``torch.to()`` / ``device_map``."""
    return backend()


def device_count() -> int:
    """Number of addressable accelerators. 0 on CPU-only hosts."""
    b = backend()
    if b == CUDA:
        torch = _torch()
        if torch is not None:
            return int(torch.cuda.device_count())
        return len(_nvidia_smi_gpus())
    return 1 if b == MPS else 0


def device_indices() -> List[int]:
    """Indices of the addressable accelerators, ``[]`` on CPU-only hosts."""
    return list(range(device_count()))


def supports_bf16() -> bool:
    """bf16 training support. MPS has no bf16 autocast path, so it is fp32 there."""
    torch = _torch()
    if torch is None or backend() != CUDA:
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def empty_cache() -> None:
    """Release cached accelerator memory. Best effort."""
    torch = _torch()
    if torch is None:
        return
    try:
        if backend() == CUDA:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif backend() == MPS:
            torch.mps.empty_cache()
    except Exception:
        pass


def probe() -> Dict[str, Any]:
    """Current accelerator status.

    Returns ``{"available", "backend", "gpus": [...]}`` where each entry has
    ``index, name, memory_total_mb, memory_used_mb, memory_free_mb,
    utilization_pct``. ``available`` is False on CPU-only hosts.
    """
    b = backend()
    if b == CUDA:
        gpus = _nvidia_smi_gpus()
        if not gpus:
            return {"available": False, "backend": b, "gpus": [], "error": "nvidia-smi unavailable"}
        return {"available": True, "backend": b, "gpus": gpus}
    if b == MPS:
        return {"available": True, "backend": b, "gpus": [_mps_device()]}
    return {"available": False, "backend": CPU, "gpus": [], "error": "no accelerator detected"}


def _nvidia_smi_gpus() -> List[Dict[str, Any]]:
    """Per-GPU metrics from ``nvidia-smi``. Empty list when it is unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    gpus: List[Dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index, name, total, used, util = parts[0], parts[1], float(parts[2]), float(parts[3]), float(parts[4])
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mb": total,
                    "memory_used_mb": used,
                    "memory_free_mb": max(0.0, total - used),
                    "utilization_pct": util,
                }
            )
        except ValueError:
            continue
    return gpus


def _mps_device() -> Dict[str, Any]:
    """Apple Silicon entry. Memory is unified, so "total" is system RAM."""
    total_mb = _system_memory_mb()
    used_mb = _mps_allocated_mb()
    return {
        "index": 0,
        "name": _apple_chip_name(),
        "memory_total_mb": total_mb,
        "memory_used_mb": used_mb,
        "memory_free_mb": max(0.0, total_mb - used_mb),
        # No per-process utilisation counter is exposed for MPS.
        "utilization_pct": None,
    }


def _system_memory_mb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _MB
    except (ValueError, OSError, AttributeError):
        return 0.0


def _mps_allocated_mb() -> float:
    torch = _torch()
    if torch is None:
        return 0.0
    try:
        return float(torch.mps.driver_allocated_memory()) / _MB
    except Exception:
        return 0.0


def _apple_chip_name() -> str:
    try:
        name = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
        ).stdout.strip()
        if name:
            return f"{name} (MPS)"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return f"{platform.machine()} (MPS)"


def select_gpu(probe_result: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Pick one device from :func:`probe` by index, falling back to the first."""
    if not (probe_result.get("available") and probe_result.get("gpus")):
        return None
    for g in probe_result["gpus"]:
        if g.get("index") == index:
            return g
    return probe_result["gpus"][0]


def free_mb_by_index() -> Dict[int, float]:
    """Free memory per device index. Empty when no accelerator is present."""
    return {int(g["index"]): float(g.get("memory_free_mb") or 0.0) for g in probe().get("gpus", [])}


def inventory_summary() -> str:
    """One-line human summary, e.g. ``A100 x2 — 60/80GB free``. Empty if unknown."""
    result = probe()
    gpus = result.get("gpus") or []
    if not gpus:
        return ""
    per = ", ".join(
        f"GPU{g['index']} {float(g.get('memory_free_mb') or 0) / 1024:.0f}/"
        f"{float(g.get('memory_total_mb') or 0) / 1024:.0f}GB free"
        for g in gpus
    )
    name = str(gpus[0].get("name") or result.get("backend"))
    prefix = f"{name} x{len(gpus)}" if len(gpus) > 1 else name
    total_free = sum(float(g.get("memory_free_mb") or 0) for g in gpus) / 1024
    return f"{prefix} — {per} (total free {total_free:.0f}GB)"
