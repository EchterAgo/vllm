# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture(autouse=True)
def _sm80_plugin_registered(monkeypatch):
    """Mark the fork's plugin as registered for these tests.

    These tests replace vLLM's plugin discovery on purpose (a patched
    ``importlib.metadata.entry_points`` or a ``VLLM_PLUGINS`` list of test
    plugins), so the fork's plugin never runs in them. The startup guard in
    ``vllm.ampere_bootstrap/guard.py`` checks real discovery and would refuse the configs
    these tests build, as would its check of the platform plugin.
    """
    monkeypatch.setattr("vllm.ampere_bootstrap._registered", True)
    monkeypatch.setattr("vllm.ampere_bootstrap.guard.check_platform", lambda: None)
