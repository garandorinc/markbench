from .cropping_attack import CroppingAttack
from .invert_colors_attack import InvertColorsAttack
from .region_blackout_attack import RegionBlackoutAttack
from .remove_lines_attack import RemoveLinesAttack

__all__ = [
    "CroppingAttack",
    "InvertColorsAttack",
    "RegionBlackoutAttack",
    "RemoveLinesAttack",
]
