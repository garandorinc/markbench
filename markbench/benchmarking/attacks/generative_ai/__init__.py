try:
    from .stable_diffusion_inpainting import (
        StableDiffusionEditAttack,
        StableDiffusionInpaintAttack,
    )
except ImportError:
    StableDiffusionEditAttack = None
    StableDiffusionInpaintAttack = None
