"""End-to-end run of the CLI with the example adapter on synthetic images."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from markbench.cli import CLI

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_IMAGES = 3


@pytest.fixture
def image_dir(tmp_path):
    directory = tmp_path / "images"
    directory.mkdir()
    rng = np.random.default_rng(0)
    for idx in range(NUM_IMAGES):
        pixels = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / f"image_{idx}.png")
    return directory


def _run(image_dir, output_dir, monkeypatch, **kwargs):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    CLI().run(
        path=str(image_dir),
        model="examples.parity_adapter:build_model",
        output=str(output_dir),
        device="cpu",
        jsonl_only=True,
        batch_size=2,
        num_workers=0,
        attack_seed=1,
        mask_seed=2,
        model_seed=3,
        **kwargs,
    )
    (run_dir,) = output_dir.iterdir()
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    return run_dir, rows


def test_cli_run_writes_one_row_per_image_and_attack(image_dir, tmp_path, monkeypatch):
    run_dir, rows = _run(
        image_dir, tmp_path / "results", monkeypatch, attacks="none,jpeg:75,horizontal_flip"
    )

    assert (run_dir / "run_config.json").exists()
    assert len(rows) == NUM_IMAGES * 3
    assert {row["attack_name"] for row in rows} == {
        "NoAttack",
        "JPEGCompressionAttack",
        "HorizontalFlipAttack",
    }

    for row in rows:
        assert 0.0 < row["mask_area"] < 1.0
        assert row["embed_metrics"]["psnr"] > 40.0
        assert 0.0 <= row["decode_metrics"]["best_message"]["bit_accuracy"] <= 1.0

    # Without an attack, the example method recovers the message and the region.
    for row in (r for r in rows if r["attack_name"] == "NoAttack"):
        assert row["decode_metrics"]["best_message"]["bit_accuracy"] == 1.0
        assert row["decode_metrics"]["iou"]["localization_iou"] > 0.8

    # A flip keeps the pixel values but moves the mask.
    for row in (r for r in rows if r["attack_name"] == "HorizontalFlipAttack"):
        assert row["attacked_mask_area"] == pytest.approx(row["mask_area"])
        assert row["attack_metrics"] == {}


def test_cli_run_accepts_attack_config_file(image_dir, tmp_path, monkeypatch):
    _, rows = _run(
        image_dir,
        tmp_path / "results",
        monkeypatch,
        attacks_config=str(REPO_ROOT / "examples" / "attacks.yaml"),
    )

    assert len(rows) == NUM_IMAGES * 5
    assert {"gaussian_blur", "splicing_untargeted"} <= {row["attack_name"] for row in rows}


def test_cli_run_requires_a_model(image_dir, tmp_path):
    with pytest.raises(ValueError, match="--model is required"):
        CLI().run(path=str(image_dir), output=str(tmp_path / "results"))
