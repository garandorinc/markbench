"""Resume helpers for local benchmark runs.

This module provides functions to:
- Save and load experiment configuration
- Validate configuration compatibility for resume
- Build skip-sets from existing results
- Compute dataset hashes for validation
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, List, Tuple

from markbench.data import ImageDataset
from markbench.benchmarking.attacks import AbstractAttack


def load_run_config(run_dir: Path) -> Optional[dict]:
    """Load existing run configuration from directory.

    Args:
        run_dir: Path to run directory containing run_config.json

    Returns:
        Configuration dictionary if found, None otherwise
    """
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config_match(existing: dict, current: dict) -> Tuple[bool, List[str]]:
    """Validate that current config matches existing config for resume.

    Args:
        existing: Previously saved configuration
        current: Current run configuration

    Returns:
        Tuple of (is_valid, list_of_differences)
    """
    differences = []

    # Model name must match exactly
    if existing.get("model_name") != current.get("model_name"):
        differences.append(
            f"Model mismatch: {existing.get('model_name')} != {current.get('model_name')}"
        )

    # Seeds must match (if they were set)
    for seed_type in ["attack_seed", "mask_seed", "model_seed"]:
        existing_seed = existing.get("seeds", {}).get(seed_type)
        current_seed = current.get("seeds", {}).get(seed_type)
        if existing_seed != current_seed:
            differences.append(
                f"{seed_type} mismatch: {existing_seed} != {current_seed}"
            )

    # Batch size must match (affects seed derivation)
    if existing.get("batch_size") != current.get("batch_size"):
        differences.append(
            f"Batch size mismatch: {existing.get('batch_size')} != {current.get('batch_size')}"
        )

    # Dataset hash should match (same images in same order)
    existing_hash = existing.get("dataset_info", {}).get("image_paths_hash")
    current_hash = current.get("dataset_info", {}).get("image_paths_hash")
    if existing_hash != current_hash:
        differences.append("Dataset changed (image paths hash mismatch)")

    # Number of images should match
    existing_num = existing.get("dataset_info", {}).get("num_images")
    current_num = current.get("dataset_info", {}).get("num_images")
    if existing_num != current_num:
        differences.append(
            f"Number of images changed: {existing_num} != {current_num}"
        )

    # Attacks should match
    existing_attacks = existing.get("attacks", [])
    current_attacks = current.get("attacks", [])
    if len(existing_attacks) != len(current_attacks):
        differences.append(
            f"Number of attacks changed: {len(existing_attacks)} != {len(current_attacks)}"
        )
    else:
        # Check each attack
        for i, (ex_atk, cur_atk) in enumerate(zip(existing_attacks, current_attacks)):
            if ex_atk != cur_atk:
                differences.append(
                    f"Attack {i} changed: {ex_atk.get('name')} != {cur_atk.get('name')}"
                )

    return (len(differences) == 0, differences)


def _normalize_config(config: dict) -> dict:
    """Normalize config dict to ensure consistent JSON serialization.

    Converts numeric types to match what attack.get_config() returns.
    This ensures skip keys match between resume and runtime.
    """
    normalized = {}
    for k, v in config.items():
        # Convert floats that are actually integers to ints
        # This matches the behavior of attack.get_config()
        if isinstance(v, float) and v.is_integer():
            normalized[k] = int(v)
        else:
            normalized[k] = v
    return normalized


def build_skip_set(results_path: Path) -> set:
    """Build set of completed (image_name, attack_name, config, seed) from JSONL.

    Args:
        results_path: Path to results.jsonl file

    Returns:
        Set of tuples (image_name, attack_name, attack_config_json, attack_seed)

    Note:
        For attacks with random parameters (e.g., GaussianBlur with sigma range),
        the combination of base config + seed uniquely identifies the result.
        The attack_params in JSONL stores the BASE config (sigma=[0.1, 2.0]),
        not resolved params, so we can directly use it for comparison.
    """
    skip_set = set()

    if not results_path.exists():
        return skip_set

    with open(results_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                attack_params = json.loads(row["attack_params"])
                # Remove post-run metadata, keep only base config
                config_params = {k: v for k, v in attack_params.items() if k != "post_run"}
                config_params = _normalize_config(config_params)
                config_json = json.dumps(config_params, sort_keys=True)

                # Get attack seed
                attack_seed = row.get("attack_seed")

                # Create unique key from image, attack, config, and seed
                # Config + seed combination uniquely identifies the result
                key = (
                    row["image_name"],
                    row["attack_name"],
                    config_json,
                    attack_seed,
                )
                skip_set.add(key)
            except json.JSONDecodeError as e:
                # Handle corrupted JSONL lines (from crashes)
                print(f"Warning: Skipping corrupted line {line_num} in results.jsonl: {e}")
                continue
            except KeyError as e:
                # Handle malformed rows missing required fields
                print(f"Warning: Skipping malformed line {line_num} in results.jsonl (missing {e})")
                continue

    return skip_set


def compute_dataset_hash(dataset: ImageDataset) -> str:
    """Compute reproducible hash of dataset for validation.

    Uses sorted image names to detect if dataset changed.

    Args:
        dataset: ImageDataset instance

    Returns:
        First 16 characters of SHA256 hash
    """
    # Collect all image names in order
    names = []
    for name, _ in dataset:
        names.append(name)

    # Hash the sorted list for reproducibility
    content = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def serialize_attacks_config(attacks: List[AbstractAttack]) -> List[dict]:
    """Convert attack instances to serializable configuration.

    Args:
        attacks: List of attack instances

    Returns:
        List of dictionaries with attack name and parameters
    """
    return [
        {"name": attack.name, "params": attack.get_config()}
        for attack in attacks
    ]
