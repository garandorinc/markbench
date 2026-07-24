from markbench.benchmarking.attacks import get_attack
from markbench.benchmarking.attacks.runtime_specs import (
    StableDiffusionMaskEditSpec,
    build_runtime_attack_spec,
)


def test_horizontal_and_vertical_flips_are_distinct():
    horizontal = get_attack("horizontal_flip")
    horizontal_alias = get_attack("hflip")
    vertical = get_attack("vertical_flip")
    vertical_alias = get_attack("vflip")

    assert horizontal.axis == "y"
    assert horizontal_alias.axis == "y"
    assert vertical.axis == "x"
    assert vertical_alias.axis == "x"


def test_new_generative_controls_are_constructible():
    variants = [{"min_mask_side_ratio": 0.2, "max_mask_side_ratio": 0.35}]

    background_preserving = build_runtime_attack_spec(
        "inpainting_off_mask_copy_content",
        {"variants": variants},
    )
    mask_outpainting = build_runtime_attack_spec(
        "mask_outpainting",
        {"variants": variants},
    )

    assert isinstance(background_preserving, StableDiffusionMaskEditSpec)
    assert background_preserving.mask_mode == "random_rect"
    assert background_preserving.copy_original_outside_attack_mask is True

    assert isinstance(mask_outpainting, StableDiffusionMaskEditSpec)
    assert mask_outpainting.mask_mode == "inverse_target_mask"
    assert mask_outpainting.copy_original_outside_attack_mask is False
