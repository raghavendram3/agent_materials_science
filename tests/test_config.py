"""Tests for configuration resolution and validation."""

import pytest

from agent_materials_science.config import (
    AgentConfig,
    DEFAULT_FAIRCHEM_MODEL,
    DEFAULT_FAIRCHEM_TASK,
    get_available_adsorbates,
)
from agent_materials_science.tools.converters import ADSORBATE_GEOMETRIES


class TestPrecedence:
    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("FAIRCHEM_MODEL", "uma-s-1p1")
        monkeypatch.setenv("FAIRCHEM_TASK", "omat")
        cfg = AgentConfig(material="Si", fairchem_model="uma-m-1p1",
                          fairchem_task="oc20")
        assert cfg.fairchem_model == "uma-m-1p1"
        assert cfg.fairchem_task == "oc20"

    def test_env_beats_default(self, monkeypatch):
        monkeypatch.setenv("FAIRCHEM_MODEL", "uma-s-1p1")
        cfg = AgentConfig(material="Si")
        assert cfg.fairchem_model == "uma-s-1p1"

    def test_default_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("FAIRCHEM_MODEL", raising=False)
        monkeypatch.delenv("FAIRCHEM_TASK", raising=False)
        cfg = AgentConfig(material="Si")
        assert cfg.fairchem_model == DEFAULT_FAIRCHEM_MODEL
        assert cfg.fairchem_task == DEFAULT_FAIRCHEM_TASK

    def test_device_from_use_gpu_alias(self, monkeypatch):
        monkeypatch.delenv("FAIRCHEM_DEVICE", raising=False)
        assert AgentConfig(material="Si", use_gpu=True).device == "cuda"
        assert AgentConfig(material="Si", use_gpu=False).device == "cpu"
        assert AgentConfig(material="Si").device == "auto"

    def test_device_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("FAIRCHEM_DEVICE", "cpu")
        assert AgentConfig(material="Si", device="cuda").device == "cuda"
        monkeypatch.delenv("FAIRCHEM_DEVICE")


class TestValidation:
    def test_invalid_device(self):
        with pytest.raises(ValueError, match="device"):
            AgentConfig(material="Si", device="tpu")

    def test_invalid_site_type(self):
        with pytest.raises(ValueError, match="site type"):
            AgentConfig(material="Si", site_types=["top", "banana"])

    def test_invalid_miller(self):
        with pytest.raises(ValueError, match="Miller"):
            AgentConfig(material="Si", miller_indices=(1, 1))

    def test_negative_fix_layers(self):
        with pytest.raises(ValueError, match="fix_layers"):
            AgentConfig(material="Si", fix_layers=-1)

    def test_mp_id_extracted_from_material(self):
        cfg = AgentConfig(material="mp-149")
        assert cfg.mp_id == "mp-149"
        assert cfg.material == ""


class TestSerialization:
    def test_from_dict_converts_lists(self):
        cfg = AgentConfig.from_dict(
            {"material": "Si", "miller_indices": [1, 1, 0], "supercell": [2, 2]}
        )
        assert cfg.miller_indices == (1, 1, 0)
        assert cfg.supercell == (2, 2)

    def test_round_trip(self):
        cfg = AgentConfig(material="Pt", adsorbate="CO", fix_layers=2)
        cfg2 = AgentConfig.from_dict(cfg.to_dict())
        assert cfg2.adsorbate == "CO"
        assert cfg2.fix_layers == 2

    def test_from_dict_does_not_mutate_input(self):
        d = {"material": "Si", "miller_indices": [1, 1, 1]}
        AgentConfig.from_dict(d)
        assert d["miller_indices"] == [1, 1, 1]


class TestAdsorbateRegistry:
    def test_single_source_of_truth(self):
        # The CLI list must expose every species the geometry table supports.
        assert set(get_available_adsorbates()) == set(ADSORBATE_GEOMETRIES)
