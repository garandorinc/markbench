# MarkBench: localized watermark robustness benchmark

MarkBench is a benchmark harness for localized invisible image watermarks. It embeds a watermark in a region of each image, applies an attack, decodes the attacked image, and logs how well the message and the region survive.

The repository contains the interfaces, dataset loaders, attacks, metrics, and runner. It does not contain watermarking models, model weights, or datasets. You connect a method through a small adapter.

## What is included

- A `WatermarkModel` interface for embedding and decoding payloads.
- Loaders for a folder of images, COCO, SA-1B, DIV2K, and MIRFLICKR-25K.
- 55 attacks: signal processing, compression, geometric, noise, local editing, splicing, and diffusion-based inpainting and outpainting.
- Metrics for message recovery (bit accuracy), localization (IoU, bounding-box IoU, Dice, coverage), image quality (PSNR, SSIM, MSE, MAE, optional LPIPS), and timing.
- A runner with deterministic seeds, resume support, and JSONL output.
- A toy example adapter in `examples/` that runs end to end without model weights.

## Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

| Extra | Adds | Command |
|---|---|---|
| `generative` | Diffusion-based inpainting and outpainting attacks | `pip install -e ".[generative]"` |
| `quality` | LPIPS metric | `pip install -e ".[quality]"` |
| `profile` | `--profile` flag | `pip install -e ".[profile]"` |
| `dev` | Tests and linter | `pip install -e ".[dev]"` |

Without the `quality` extra, the runner omits the `lpips` field from the results.

The diffusion attacks download the Stable Diffusion inpainting weights (`runwayml/stable-diffusion-inpainting`) from Hugging Face on first use. Those weights have their own license (CreativeML OpenRAIL-M).

### Tested versions

`pyproject.toml` gives lower bounds only. The benchmark was developed with these versions:

| Package | Version |
|---|---|
| torch | 2.9.1 |
| torchvision | 0.24.1 |
| kornia | 0.8.2 |
| torchmetrics | 1.8.2 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0.86 |
| Pillow | 12.1.0 |
| scipy | 1.17.0 |
| PyWavelets | 1.9.0 |
| diffusers | 0.37.0 |
| lpips | 0.1.4 |

## Quick start

The example adapter needs no weights. Run it from the repository root:

```bash
markbench run \
  --path /path/to/images \
  --model examples.parity_adapter:build_model \
  --attacks "none,jpeg:75,horizontal_flip" \
  --output results \
  --jsonl_only True \
  --max_images 8
```

The example method is not robust. It recovers the message with no attack and fails under most attacks. Its purpose is to show the adapter contract.

## Add a watermarking method

Write an adapter that subclasses `markbench.interface.WatermarkModel`, and a builder function that returns an instance. See `examples/parity_adapter.py` for a complete adapter.

```python
# my_method_adapter.py
from markbench.interface import DecodeResult, EmbedResult, WatermarkData, WatermarkModel


class MyWatermark(WatermarkModel):
    @property
    def name(self):
        return "my_method"

    @property
    def nbits(self):
        return 32

    def embed(self, image, watermark_data):
        watermarked = ...
        return EmbedResult(watermarked=watermarked, watermark_data=watermark_data)

    def decode(self, image):
        message, localization_map = ...
        return DecodeResult(
            watermark_data=WatermarkData(message=message, localization_map=localization_map)
        )


def build_model(device="cuda", **kwargs):
    return MyWatermark(...)
```

The runner calls the adapter with batches:

| Call | Argument | Shape and range |
|---|---|---|
| `embed` | `image` | `(B, 3, H, W)`, float, values in [0, 255] |
| `embed` | `watermark_data.message` | `(B, N_masks, nbits)`, int64, values in {0, 1} |
| `embed` | `watermark_data.localization_map` | `(B, N_masks, H, W)`, float, values in {0, 1} |
| `decode` | `image` | `(B, 3, H', W')`, float, values in [0, 255] |

`embed` returns the watermarked images with the same shape and range as the input. `decode` returns a message of shape `(B, N_decoded, nbits)` and a localization map of shape `(B, N_decoded, H', W')`. The decode metrics need the localization map. A method without localization can return a map of ones.

Rules that the runner applies:

- The runner generates the message from `--model_seed`. The adapter does not select the message.
- `N_masks` is 1 unless the adapter sets `supports_multiple_messages` to `True`.
- A model whose `name` is `trustmark`, `tm`, or `trust_mark` receives rectangular masks only. The runner converts each mask to its bounding box for such a model.

Pass the builder to the CLI as `module:function`. The CLI adds the working directory to the import path. Use `--model_kwargs` to send extra arguments to the builder:

```bash
markbench run \
  --path /path/to/images \
  --model my_method_adapter:build_model \
  --model_kwargs '{"checkpoint": "weights.pth"}' \
  --attacks "none,jpeg:75" \
  --output results
```

## Datasets

The runner resizes each image to 512 x 512 before embedding.

For a folder of images, pass `--path` only. The runner generates one random mask per image. The target area of the mask is 4% to 14% of the image.

For a built-in loader, pass `--dataset` and the local root in `--path`:

| `--dataset` | Expected layout under `--path` | Masks |
|---|---|---|
| `coco` | `train2017/`, `annotations/instances_train2017.json` | Annotation masks |
| `sa1b` | `sa_000000/` to `sa_000009/`, one `.json` file per image | Annotation masks |
| `div2k` | `DIV2K_train_HR/`, `DIV2K_valid_HR/` | Random masks |
| `mirflickr` | `mirflickr/` | Random masks |

For annotation masks, `--annotation_min_area` sets the minimum mask area as a fraction of the image. The default is 0.02. The runner skips an image that has no mask above this limit.

```bash
markbench run \
  --dataset coco \
  --path /path/to/coco2017 \
  --model my_method_adapter:build_model \
  --attacks "horizontal_flip,vertical_flip" \
  --output results \
  --jsonl_only True
```

## Attacks

List the attack names:

```bash
markbench list attacks
```

`--attacks` takes a comma-separated list. Parameters follow the name after a colon:

```bash
--attacks "none,jpeg:75,gaussian_blur:5,rotation:15"
```

`chain(a,b)` applies attacks in sequence. `random(2,a,b,c)` applies 2 attacks from the pool.

### Attack files

`--attacks_config` reads the attacks from a YAML file. See `examples/attacks.yaml`. An entry with a `variants` list is resolved per image: the runner selects one variant for each image from the attack seed.

These attacks need the dataset or the watermark mask at run time. Use them through `--attacks_config`:

| Name | Description |
|---|---|
| `inpainting_untargeted` | Inpainting of a random rectangle |
| `inpainting_targeted` | Inpainting of the watermark mask |
| `inpainting_off_mask_copy_content` | Inpainting of a random rectangle. The runner then copies the watermarked pixels outside the rectangle back into the output. This isolates the edited region from diffusion changes in the rest of the image. |
| `mask_outpainting` | Outpainting that keeps the watermark mask as the retained object and edits the region outside it. The image size does not change. |
| `splicing_untargeted` | Splicing of content from a second image of the same dataset |
| `splicing_targeted` | Splicing of content from a second image of the same dataset over the watermark mask |

```yaml
attacks:
  - name: inpainting_off_mask_copy_content
    variants:
      - min_mask_side_ratio: 0.20
        max_mask_side_ratio: 0.35
    prompt: ""
    guidance_scale: 7.5
    num_inference_steps: 25
  - name: mask_outpainting
    variants:
      - min_mask_side_ratio: 0.20
        max_mask_side_ratio: 0.35
    prompt: ""
    guidance_scale: 7.5
    num_inference_steps: 25
```

### Sample attack files

`configs/` contains attack files that you can pass to `--attacks_config`:

| File | Content |
|---|---|
| `configs/attacks_test.yaml` | 4 basic attacks, for a quick check |
| `configs/ihmmsec_compact53.yaml` | A compact grid of 53 attacks. Each attack except one has a list of variants. It includes the diffusion and splicing attacks. |
| `configs/by_papers/attacks_watermark_anything.yaml` | Attack settings from the Watermark Anything paper |
| `configs/by_papers/attacks_trustmark.yaml` | Attack settings from the TrustMark paper |
| `configs/by_papers/attacks_maskwm.yaml` | Attack settings from the MaskWM paper |
| `configs/by_papers/attacks_omniguard.yaml` | Attack settings from the OmniGuard paper |
| `configs/by_papers/attacks_editguard.yaml` | Attack settings from the EditGuard paper |
| `configs/qualitative_edit_smoke.yaml` | Inpainting and splicing, targeted and untargeted |
| `configs/qualitative_splicing_smoke.yaml` | Splicing with each mask mode |

The files in `configs/by_papers/` give the paper reference in the header.

```bash
markbench run \
  --path /path/to/images \
  --model my_method_adapter:build_model \
  --attacks_config configs/by_papers/attacks_watermark_anything.yaml \
  --attack_seed 0 --mask_seed 0 --model_seed 0 \
  --output results \
  --jsonl_only True
```

### Flips

- `horizontal_flip`: left-right flip.
- `vertical_flip`: top-bottom flip.
- `diagonal_flip`: transpose-like diagonal flip.

## Output

Each run writes to `<output>/<model name>_<timestamp>/`:

| File | Content |
|---|---|
| `results.jsonl` | One row per image and attack |
| `run_config.json` | Model, dataset hash, attacks, seeds, and batch size |
| `timing.json` | Total run time |
| `images/` | Original, watermarked, and attacked images. `--jsonl_only True` disables this. |

Each row in `results.jsonl` contains the image name, the attack name and parameters, the embed, attack, and decode metrics, the mask statistics, and the seeds. For an attack given through `--attacks`, the `attack_name` field is the class name of the attack, for example `JPEGCompressionAttack`. For an entry with `variants`, it is the name from the YAML file.

## Reproducibility

- `--attack_seed`, `--mask_seed`, and `--model_seed` make the attack parameters, the masks, and the messages deterministic. Without a seed, the related step is random.
- The seeds of a batch depend on the batch index. Use the same `--batch_size` to reproduce a run.
- `--resume <run directory>` continues a run. The configuration must match `run_config.json`.
- `--attack_split_worker_count` and `--attack_split_worker_id` divide the attack list between parallel jobs.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests run the CLI end to end with the example adapter on synthetic images. They also check the attack registry and parse each file in `configs/`. They do not run a diffusion model.

## License

Copyright (c) 2026 Garandor Limited.

The code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0). See `LICENSE`.
