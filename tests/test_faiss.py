import sys
import types

import pytest

from fiser.faiss_utils import import_faiss_gpu


def test_faiss_guard_rejects_cpu_only_build(monkeypatch):
    cpu_only = types.ModuleType("faiss")
    cpu_only.get_num_gpus = lambda: 0
    monkeypatch.setitem(sys.modules, "faiss", cpu_only)

    with pytest.raises(RuntimeError, match="no visible CUDA device"):
        import_faiss_gpu()
