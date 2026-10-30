# SPDX-License-Identifier: Apache-2.0
"""CPU-only event contract tests; real stream ordering needs GPU validation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


class TestOutputEvent(unittest.TestCase):
    def setUp(self):
        self.cuda = Mock()

        class OmniPlatform:
            @classmethod
            def record_device_event(cls):
                return None

        # Load the real platform class without initializing GPU/model plugins.
        dependencies = {
            "torch": {"cuda": self.cuda},
            "vllm.logger": {"init_logger": Mock()},
            "vllm.platforms.interface": {"DeviceCapability": object},
            "vllm_metax.platform": {"MacaPlatform": type("MacaPlatform", (), {})},
            "vllm_omni.platforms.interface": {
                "OmniPlatform": OmniPlatform,
                "OmniPlatformEnum": SimpleNamespace(CUDA=0),
            },
        }
        modules = {name: ModuleType(name) for name in dependencies}
        for name, attributes in dependencies.items():
            modules[name].__dict__.update(attributes)
        path = Path(__file__).resolve().parents[2] / "src/vllm_omni_metax/platform.py"
        spec = importlib.util.spec_from_file_location("_metax_output_event_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        self.platform = module.MetaxOmniPlatform

    def test_records_current_stream_without_device_sync(self):
        event = self.cuda.Event.return_value
        self.assertIs(self.platform.record_device_event(), event)
        event.record.assert_called_once_with(self.cuda.current_stream.return_value)
        self.cuda.synchronize.assert_not_called()

    def test_event_failure_falls_back_to_device_sync(self):
        for operation in (self.cuda.Event, self.cuda.Event.return_value.record):
            with self.subTest(operation=operation):
                self.cuda.reset_mock(side_effect=True)
                operation.side_effect = RuntimeError("event failed")
                self.assertIsNone(self.platform.record_device_event())
                self.cuda.synchronize.assert_called_once_with()

    def test_sync_failure_propagates(self):
        self.cuda.Event.side_effect = RuntimeError("event failed")
        self.cuda.synchronize.side_effect = RuntimeError("sync failed")
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            self.platform.record_device_event()


if __name__ == "__main__":
    unittest.main()
