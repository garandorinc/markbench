"""Command-line interface for local watermark robustness evaluation."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import fire

from markbench.data.image_dataset import MaskMode


class ListCommands:
    """List available resources."""

    def models(self):
        """List available watermarking models."""
        from markbench.models import list_models

        models = list_models()
        print(f"Models ({len(models)}):")
        for m in models:
            print(f"  - {m}")

    def attacks(self):
        """List available attacks."""
        from markbench.benchmarking.attacks import list_attacks

        attacks = list_attacks()
        print(f"Attacks ({len(attacks)}):")
        for a in attacks:
            print(f"  - {a}")

    def datasets(self):
        """List available datasets."""
        from markbench.data import list_datasets

        datasets = list_datasets()
        print(f"Datasets ({len(datasets)}):")
        for d in datasets:
            print(f"  - {d}")


class CLI:
    """Watermark robustness benchmark CLI."""

    def __init__(self):
        self.list = ListCommands()

    def run(
        self,
        path: str = "",
        dataset: str = "",
        model: str = "",
        model_kwargs: dict = None,
        attacks: str = "none,jpeg:50,jpeg:75,jpeg:90",
        attacks_config: str = "",
        output: str = "results",
        resume: str = "",
        device: str = "",
        max_images: int = None,
        jsonl_only: bool = False,
        visualize_masks: bool = False,
        save_masks: bool = False,
        log_messages: bool = True,
        attack_seed: int = None,
        mask_seed: int = None,
        model_seed: int = None,
        num_workers: int = 4,
        batch_size: int = 4,
        profile: bool = False,
        attack_split_worker_count: int = 1,
        attack_split_worker_id: int = 0,
        round_before_quantize: bool = True,
        annotation_min_area: float = 0.02,
    ):
        """Run watermark benchmark.

        Args:
            path: Path to image/directory (uses SimpleImageDataset if --dataset not set, otherwise overrides dataset's default root)
            dataset: Dataset name to use (uses default root if --path not set)
            model: Registered model name, or a "module:function" builder that returns a WatermarkModel
            model_kwargs: Extra keyword arguments for the model builder, e.g. '{"checkpoint": "weights.pth"}'
            attacks: Comma-separated attack specs (e.g., "jpeg:50,blur:3")
            attacks_config: Path to YAML attack config (overrides --attacks if set)
            output: Output directory for results
            resume: Path to existing run directory to resume (e.g., "results/my_method_20260101_120000")
            device: Device (default: auto-detect)
            max_images: Limit number of images (for testing)
            jsonl_only: Only write results.jsonl, skip image artifacts
            visualize_masks: Generate mask transformation visualizations (default: False)
            save_masks: Save mask tensors to .pt files (default: False)
            log_messages: Save original and decoded messages in metrics (default: True)
            attack_seed: Base seed for attack randomness (optional)
            mask_seed: Base seed for random mask generation (optional)
            model_seed: Base seed for model message generation (optional)
            num_workers: Number of worker threads for parallel I/O (default: 4)
            batch_size: Batch size for processing (default: 4)
            profile: Enable profiling and generate flamegraph (default: False). Needs the "profile" extra.
            attack_split_worker_count: Total number of workers for attack distribution (default: 1)
            attack_split_worker_id: 0-indexed ID of this worker (default: 0)
            round_before_quantize: Round values before uint8 conversion (default: True). If False, truncate (floor).
            annotation_min_area: Minimum annotation mask area ratio for COCO/SA-1B selection.
                                 Masks smaller than this threshold are ignored. Images
                                 without any qualifying annotation mask are skipped.
        """
        from dotenv import load_dotenv
        from markbench.utils.device import get_device
        from markbench.data import SimpleImageDataset, DATASET_REGISTRY, list_datasets
        from markbench.models import load_model
        from markbench.benchmarking.attacks import parse_attack_string, parse_attack_config, _split_preserving_parens
        from markbench.logger import BenchmarkLogger
        from markbench.runner import run_benchmark
        # Load environment variables (e.g. HF_TOKEN, HF_HOME) from a local .env file
        load_dotenv()

        if not model:
            raise ValueError(
                "--model is required. Pass a registered model name or a "
                "module:function builder, e.g. --model my_method_adapter:build_model"
            )

        if not path and not dataset:
            raise ValueError(
                "At least one of --path or --dataset must be provided. "
                f"Available datasets: {list_datasets()}"
            )

        # Validate worker splitting parameters
        if attack_split_worker_count < 1:
            raise ValueError("--attack-split-worker-count must be >= 1")
        if attack_split_worker_id < 0:
            raise ValueError("--attack-split-worker-id must be >= 0")
        if attack_split_worker_id >= attack_split_worker_count:
            raise ValueError(
                f"--attack-split-worker-id ({attack_split_worker_id}) must be < "
                f"--attack-split-worker-count ({attack_split_worker_count})"
            )
        if not 0.0 <= annotation_min_area <= 1.0:
            raise ValueError("--annotation_min_area must be in [0, 1]")

        device = get_device(device)
        print(f"Using device: {device}")

        if dataset:
            if dataset not in DATASET_REGISTRY:
                raise ValueError(
                    f"Unknown dataset: '{dataset}'. "
                    f"Available datasets: {list_datasets()}"
                )
            dataset_config = DATASET_REGISTRY[dataset]
            DatasetClass = dataset_config["class"]
            root = path if path else dataset_config["default_root"]
            ds = DatasetClass(root=root, max_samples=max_images, mask_mode=MaskMode.ANNOTATION
                              if dataset in ["coco", "sa1b"]
                              else MaskMode.NONE)
        else:
            data_path = Path(path)
            if data_path.is_file():
                ds = SimpleImageDataset(str(data_path.parent), max_samples=1)
                ds.image_paths = [data_path]
            else:
                ds = SimpleImageDataset(path, max_samples=max_images)

        print(f"Found {len(ds)} images")

        # Load model
        print(f"Loading model: {model}")
        model_kwargs = dict(model_kwargs or {})
        # A console script does not put the working directory on sys.path.
        # Add it so that a local adapter module is importable.
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        wm_model = load_model(model, device=device, **model_kwargs)

        # Parse attacks
        if attacks_config:
            import yaml

            with open(attacks_config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            attack_list = parse_attack_config(config, device=device)
        elif attacks.strip().lower() == "all":
            from markbench.benchmarking.attacks import get_all_attacks

            attack_list = get_all_attacks()
        else:
            attack_list = [parse_attack_string(a.strip()) for a in _split_preserving_parens(attacks)]
        print(f"Attacks: {[a.name for a in attack_list]}")

        # Keep reference to full attack list for resume config
        original_attack_list = attack_list

        # Filter attacks for this worker using round-robin distribution
        if attack_split_worker_count > 1:
            # Create mapping: (global_attack_idx, attack)
            # (Preserves global indices for proper seeding)
            attack_index_mapping = [
                (global_idx, attack)
                for global_idx, attack in enumerate(original_attack_list)
                if global_idx % attack_split_worker_count == attack_split_worker_id
            ]
            attack_list = [attack for _, attack in attack_index_mapping]

            print(f"Worker {attack_split_worker_id}/{attack_split_worker_count}: "
                  f"Processing {len(attack_list)} of {len(original_attack_list)} attacks")
            print(f"Attack indices: {[idx for idx, _ in attack_index_mapping]}")
        else:
            # Single worker mode - trivial mapping
            attack_index_mapping = [(idx, attack) for idx, attack in enumerate(attack_list)]

        # Handle resume or new run
        skip_set = None
        run_name = None

        if resume:
            resume_dir = Path(resume)

            if not resume_dir.exists():
                raise ValueError(f"Resume directory not found: {resume}")

            from markbench.resume import (
                load_run_config,
                validate_config_match,
                build_skip_set,
                compute_dataset_hash,
                serialize_attacks_config
            )

            # Load existing config
            existing_config = load_run_config(resume_dir)
            if existing_config is None:
                raise ValueError(
                    f"Cannot resume: {resume} does not contain run_config.json. "
                    "This directory may be from an old version of markbench."
                )

            # Build current config
            current_config = {
                "model_name": wm_model.name,
                "dataset_info": {
                    "type": ds.__class__.__name__,
                    "num_images": len(ds),
                    "image_paths_hash": compute_dataset_hash(ds),
                },
                "attacks": serialize_attacks_config(original_attack_list),
                "seeds": {
                    "attack_seed": attack_seed,
                    "mask_seed": mask_seed,
                    "model_seed": model_seed,
                },
                "batch_size": batch_size,
                "annotation_min_area": annotation_min_area,
                "model_kwargs": model_kwargs,
                "worker_split": {
                    "worker_count": attack_split_worker_count,
                    "worker_id": attack_split_worker_id,
                },
            }

            # Worker split configuration must match
            existing_split = existing_config.get("worker_split", {"worker_count": 1, "worker_id": 0})
            current_split = current_config.get("worker_split", {"worker_count": 1, "worker_id": 0})
            if existing_split != current_split:
                differences = [
                    f"Worker split configuration changed: {existing_split} != {current_split}"
                ]
                error_msg = "Cannot resume: configuration mismatch:\n" + "\n".join(
                    f"  - {diff}" for diff in differences
                )
                raise ValueError(error_msg)

            # Validate match
            is_valid, differences = validate_config_match(existing_config, current_config)
            if not is_valid:
                error_msg = "Cannot resume: configuration mismatch:\n" + "\n".join(
                    f"  - {diff}" for diff in differences
                )
                raise ValueError(error_msg)

            # Build skip set
            skip_set = build_skip_set(resume_dir / "results.jsonl")
            print(f"Resuming from {resume}")
            print(f"Found {len(skip_set)} completed image-attack combinations")

            # Reuse same directory and run name
            output_dir = resume_dir.parent
            run_name = resume_dir.name

        else:
            # New run - generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if attack_split_worker_count > 1:
                run_name = f"{wm_model.name}_{timestamp}_worker_{attack_split_worker_id}"
            else:
                run_name = f"{wm_model.name}_{timestamp}"
            output_dir = Path(output)

        # Setup logger
        logger = BenchmarkLogger(
            output_dir,
            run_name,
            save_images=not jsonl_only,
            save_visualizations=visualize_masks,
            save_masks=save_masks,
            log_messages=log_messages,
            num_workers=num_workers,
            round_before_quantize=round_before_quantize,
        )

        # Save config for new runs (enables future resume)
        if not resume:
            from markbench.resume import compute_dataset_hash, serialize_attacks_config

            run_config = {
                "model_name": wm_model.name,
                "dataset_info": {
                    "type": ds.__class__.__name__,
                    "num_images": len(ds),
                    "image_paths_hash": compute_dataset_hash(ds),
                },
                "attacks": serialize_attacks_config(original_attack_list),
                "seeds": {
                    "attack_seed": attack_seed,
                    "mask_seed": mask_seed,
                    "model_seed": model_seed,
                },
                "batch_size": batch_size,
                "annotation_min_area": annotation_min_area,
                "model_kwargs": model_kwargs,
                "worker_split": {
                    "worker_count": attack_split_worker_count,
                    "worker_id": attack_split_worker_id,
                },
                "created_at": datetime.now().isoformat(),
            }
            logger.save_run_config(run_config)

        # Run benchmark
        print("Running benchmark...")
        start_time = time.time()

        # Initialize profiler if requested
        if profile:
            from pyinstrument import Profiler
            profiler = Profiler()
            profiler.start()

        run_benchmark(
            ds,
            wm_model,
            attack_list,
            logger,
            device=device,
            attack_seed=attack_seed,
            mask_seed=mask_seed,
            model_seed=model_seed,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_set=skip_set,
            attack_global_indices=attack_index_mapping,
            annotation_min_area=annotation_min_area,
        )

        end_time = time.time()

        # Wait for all pending writes to complete
        logger.close()
        logger.log_timing(end_time - start_time)

        # Save flamegraph if profiling was enabled
        if profile:
            profiler.stop()
            logger.save_profile(profiler)

        print(f"Results saved to: {logger.json_path}")


def main():
    """Entry point for markbench CLI."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
