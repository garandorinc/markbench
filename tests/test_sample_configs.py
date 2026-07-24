"""Each sample attack file in configs/ must parse with the current attack registry."""

from pathlib import Path

import pytest
import yaml

from markbench.benchmarking.attacks import parse_attack_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.rglob("*.yaml"))

# These names build the diffusion pipeline in the constructor. The test does
# not download model weights, so it skips them.
DIFFUSION_ATTACKS = {"sd_inpaint", "inpainting", "outpainting", "sd_outpaint"}


def _is_diffusion_entry(entry) -> bool:
    return isinstance(entry, dict) and str(entry.get("name", "")).lower() in DIFFUSION_ATTACKS


def test_sample_configs_exist():
    assert CONFIG_FILES


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: str(p.relative_to(CONFIG_DIR)))
def test_sample_config_parses(config_path):
    entries = yaml.safe_load(config_path.read_text())["attacks"]
    parseable = [entry for entry in entries if not _is_diffusion_entry(entry)]

    attacks = parse_attack_config({"attacks": parseable})

    assert len(attacks) == len(parseable) > 0
