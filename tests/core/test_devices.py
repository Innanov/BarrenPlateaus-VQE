"""Unit tests for the device factory.

Checks that each supported backend builds a working device, that the default is
lightning.qubit, and that lightning.gpu falls back to lightning.qubit when the GPU
backend cannot be created.

Run with: pytest tests/core/test_devices.py
"""

import pennylane as qml
import pytest
from pennylane import numpy as pnp

from src.core import devices
from src.core.devices import BACKENDS, active_backend, make_device


def _expval(dev):
    """Run a tiny circuit on the device and return an expectation value."""

    @qml.qnode(dev)
    def circuit(x):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    return float(circuit(pnp.array(0.7)))


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Reset the per-backend resolution cache so each test probes fresh."""
    devices._resolved.clear()
    yield
    devices._resolved.clear()


def test_backends_listed():
    assert BACKENDS == ("lightning.qubit", "default.qubit", "lightning.gpu")


def test_default_is_lightning_qubit():
    assert active_backend() == "lightning.qubit"


@pytest.mark.parametrize("backend", ["lightning.qubit", "default.qubit"])
def test_cpu_backends_build_and_run(backend):
    # CPU simulators ship with PennyLane, so they resolve to themselves and run.
    assert active_backend(backend) == backend
    dev = make_device(2, backend=backend)
    assert _expval(dev) == pytest.approx(0.764842, abs=1e-5)


def test_cpu_backends_agree():
    # The two CPU simulators must give the same physics.
    a = _expval(make_device(2, backend="lightning.qubit"))
    b = _expval(make_device(2, backend="default.qubit"))
    assert a == pytest.approx(b)


def test_gpu_falls_back_when_unavailable(monkeypatch):
    # Force the GPU probe to fail; it must fall back to lightning.qubit.
    real_device = qml.device

    def fake_device(name, *args, **kwargs):
        if name == "lightning.gpu":
            raise RuntimeError("no GPU here")
        return real_device(name, *args, **kwargs)

    monkeypatch.setattr(qml, "device", fake_device)
    assert active_backend("lightning.gpu") == "lightning.qubit"


def test_gpu_used_when_available(monkeypatch):
    # If the GPU probe succeeds, lightning.gpu is used as requested.
    real_device = qml.device

    def fake_device(name, *args, **kwargs):
        target = "lightning.qubit" if name == "lightning.gpu" else name
        return real_device(target, *args, **kwargs)

    monkeypatch.setattr(qml, "device", fake_device)
    assert active_backend("lightning.gpu") == "lightning.gpu"
