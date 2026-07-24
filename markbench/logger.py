"""Local JSONL logger for benchmark runs."""

from __future__ import annotations

import atexit
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class BenchmarkRow:
    """Single benchmark result row for one image and one attack."""

    image_name: str
    attack_name: str
    attack_params: str

    embed_time_ms: float
    embed_metrics: dict[str, float]

    attack_time_ms: float
    attack_metrics: dict[str, float]

    decode_time_ms: float
    decode_metrics: dict[str, Any]

    created_at: str

    original_path: Optional[str] = None
    watermarked_path: Optional[str] = None
    attacked_path: Optional[str] = None
    visualization_path: Optional[str] = None
    visualization_decode_path: Optional[str] = None

    mask_paths: Optional[dict[str, str]] = None

    mask_area: Optional[float] = None
    mask_bbox_area: Optional[float] = None
    mask_bbox: Optional[str] = None
    attacked_mask_area: Optional[float] = None
    attacked_mask_bbox_area: Optional[float] = None
    attacked_mask_bbox: Optional[str] = None

    attack_seed: Optional[int] = None
    mask_seed: Optional[int] = None
    model_seed: Optional[int] = None


class BenchmarkLogger:
    """Local JSONL logger with optional image, mask, and profile outputs."""

    def __init__(
        self,
        output_dir: Path,
        run_name: str,
        save_images: bool = True,
        save_visualizations: bool = False,
        save_masks: bool = False,
        log_messages: bool = True,
        num_workers: int = 4,
        round_before_quantize: bool = True,
    ):
        self.run_dir = Path(output_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._json_path = self.run_dir / "results.jsonl"
        self.save_images = save_images
        self.save_visualizations = save_visualizations
        self.save_masks = save_masks
        self.log_messages = log_messages
        self.round_before_quantize = round_before_quantize

        self._executor = ThreadPoolExecutor(max_workers=max(1, int(num_workers)))
        self._futures = []
        self._error_log = self.run_dir / "errors.jsonl"
        self._closed = False
        self._write_errors: list[str] = []
        atexit.register(self._cleanup)

        if save_images:
            (self.run_dir / "images" / "original").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "images" / "watermarked").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "images" / "attacked").mkdir(parents=True, exist_ok=True)
        if save_visualizations:
            (self.run_dir / "images" / "visualizations").mkdir(parents=True, exist_ok=True)
        if save_masks:
            (self.run_dir / "masks").mkdir(parents=True, exist_ok=True)

    def log(self, row: BenchmarkRow) -> None:
        """Queue a JSONL write."""
        self._futures.append(self._executor.submit(self._write_log, row))

    def _write_log(self, row: BenchmarkRow) -> None:
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), allow_nan=True, ensure_ascii=False) + "\n")

    def save_image(self, tensor: torch.Tensor, category: str, name: str) -> Optional[str]:
        """Queue an image save and return the relative path in the run directory."""
        if not self.save_images:
            return None

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.run_dir / "images" / category / f"{name}_{timestamp}.png"
        rel_path = str(path.relative_to(self.run_dir))
        self._futures.append(self._executor.submit(self._save_image_worker, tensor, path))
        return rel_path

    def _save_image_worker(self, tensor: torch.Tensor, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        from markbench.utils.image import save_image

        save_image(tensor, path, round_before_quantize=self.round_before_quantize)

    def save_tensor(self, tensor: torch.Tensor, name: str) -> Optional[str]:
        """Queue a tensor save and return the relative path in the run directory."""
        if not self.save_masks:
            return None

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.run_dir / "masks" / f"{name}_{timestamp}.pt"
        rel_path = str(path.relative_to(self.run_dir))
        self._futures.append(self._executor.submit(self._save_tensor_worker, tensor, path))
        return rel_path

    def _save_tensor_worker(self, tensor: torch.Tensor, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)

    def close(self) -> None:
        """Wait for all pending writes and shut down worker threads."""
        if self._closed:
            return
        self._closed = True

        for future in self._futures:
            try:
                future.result()
            except Exception as exc:
                message = f"Warning: Write operation failed: {exc}"
                print(message, file=sys.stderr)
                self._log_error(str(exc))
                self._write_errors.append(str(exc))

        self._executor.shutdown(wait=True)
        self._futures.clear()

        if self._write_errors:
            unique_errors = list(dict.fromkeys(self._write_errors))
            preview = "; ".join(unique_errors[:3])
            raise RuntimeError(
                f"BenchmarkLogger encountered {len(self._write_errors)} async write failure(s): {preview}"
            )

    def _cleanup(self) -> None:
        if not self._closed:
            self.close()

    def _log_error(self, error_message: str) -> None:
        try:
            with open(self._error_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": datetime.now().isoformat(), "error": error_message}) + "\n")
        except Exception:
            pass

    def save_run_config(self, config: dict) -> None:
        with open(self.run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def log_timing(self, duration_seconds: float) -> None:
        with open(self.run_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump({"total_duration_seconds": duration_seconds}, f)

    def save_profile(self, profiler) -> None:
        profile_path = self.run_dir / "profile.html"
        with open(profile_path, "w", encoding="utf-8") as f:
            f.write(profiler.output_html())
        print(f"Profile saved to: {profile_path}")

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def write_errors(self) -> list[str]:
        return list(self._write_errors)
