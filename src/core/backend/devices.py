"""Central PennyLane device factory.

All QNodes go through make_device, so the simulator backend is chosen in one
place. Three backends are supported:

    lightning.qubit  fast C++ statevector simulator (default)
    default.qubit    pure-Python simulator
    lightning.gpu    GPU statevector simulator (Linux + plugin only)

Pick one with the backend keyword, e.g. make_device(4, backend="default.qubit").
When backend is omitted it defaults to lightning.qubit.

The CPU simulators ship with PennyLane and are used directly. lightning.gpu is the
only backend that can be missing, so it is probed once and falls back to
lightning.qubit if it cannot be created.

lightning.gpu needs Linux (or WSL2), a CUDA GPU, and the pennylane-lightning-gpu
plugin (pip install pennylane-lightning-gpu). It is not available on native
Windows, so requesting it there always falls back to lightning.qubit.
"""

from __future__ import annotations

import pennylane as qml

BACKENDS = ("lightning.qubit", "default.qubit", "lightning.gpu")

_DEFAULT_BACKEND = "lightning.qubit"
_GPU_FALLBACK = "lightning.qubit"

# Cache of GPU probes, keyed by backend name, so each is probed only once.
_resolved: dict[str, str] = {}


def _resolve_backend(backend: str) -> str:
    """Return the usable backend, probing only lightning.gpu for availability."""
    if backend in _resolved:
        return _resolved[backend]

    usable = backend
    if backend == "lightning.gpu":
        try:
            qml.device(backend, wires=1)
        except Exception:
            usable = _GPU_FALLBACK

    _resolved[backend] = usable
    return usable


def make_device(n_qubits: int, backend: str | None = None):
    """Create a PennyLane device on n_qubits wires.

    Args:
        n_qubits: Number of wires.
        backend: One of BACKENDS. Defaults to lightning.qubit.
    """
    return qml.device(_resolve_backend(backend or _DEFAULT_BACKEND), wires=n_qubits)


def active_backend(backend: str | None = None) -> str:
    """Return the resolved backend name (for logging/provenance)."""
    return _resolve_backend(backend or _DEFAULT_BACKEND)
