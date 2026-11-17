# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Omni 0.32 deploy config factory contract."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


FACTORY_MODULE = "vllm_omni.config.config_factory"
DEPLOY_PATCH = (
    Path(__file__).resolve().parents[2]
    / "src/vllm_omni_metax/patches/deploy_resolution_patch.py"
)


class DeployResolutionPatchTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("metax_deploy_patch_under_test", DEPLOY_PATCH)
        self.deploy_patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.deploy_patch)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.deploy_patch._DEPLOY_DIR = Path(temporary.name)
        self.plugin_yaml = self.deploy_patch._DEPLOY_DIR / "qwen3_omni.yaml"
        self.plugin_yaml.touch()

        class StageConfigFactory:
            @classmethod
            def try_infer_model_type(cls, model, trust_remote_code):
                return model

            @classmethod
            def create_from_model(
                cls, model, *, trust_remote_code, cli_overrides, deploy_config_path, strategy_specs=None
            ):
                return deploy_config_path, strategy_specs

            create_legacy_stage_configs_from_model = create_from_model

        self.factory = StageConfigFactory
        self.factory_module = ModuleType(FACTORY_MODULE)

    def test_forwards_strategy_specs_and_preserves_deploy_precedence(self):
        self.factory_module.StageConfigFactory = self.factory
        strategies = {"sequence_parallel": {"ulysses_degree": 4}}
        with patch.dict(sys.modules, {FACTORY_MODULE: self.factory_module}):
            self.deploy_patch.apply_deploy_resolution_patch()
            for create, configured, expected in (
                (self.factory.create_from_model, None, str(self.plugin_yaml)),
                (self.factory.create_legacy_stage_configs_from_model, "explicit.yaml", "explicit.yaml"),
            ):
                with self.subTest(create=create.__name__):
                    deploy_path, forwarded_strategies = create(
                        "qwen3_omni",
                        trust_remote_code=None,
                        cli_overrides={},
                        deploy_config_path=configured,
                        strategy_specs=strategies,
                    )
                    self.assertEqual(deploy_path, expected)
                    self.assertIs(forwarded_strategies, strategies)

    def test_missing_factory_defers_without_import_and_installs_once(self):
        original = self.factory.create_from_model
        with patch.dict(sys.modules):
            sys.modules.pop(FACTORY_MODULE, None)
            with patch("builtins.__import__", side_effect=AssertionError("Unexpected import")):
                self.deploy_patch.apply_deploy_resolution_patch()
            self.assertEqual(original, self.factory.create_from_model)

            self.factory_module.StageConfigFactory = self.factory
            sys.modules[FACTORY_MODULE] = self.factory_module
            self.deploy_patch.ensure_patch_installed()
            installed = self.factory.create_from_model
            self.assertNotEqual(original, installed)

            self.deploy_patch.ensure_patch_installed()
            self.assertEqual(installed, self.factory.create_from_model)


if __name__ == "__main__":
    unittest.main()
