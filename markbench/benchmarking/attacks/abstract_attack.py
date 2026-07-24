from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Any
import torch

if TYPE_CHECKING:
    from markbench.interface import AttackResult


class AbstractAttack(ABC):
    """
    Abstract base class for image attacks.

    Attacks operate on torch.Tensor images in (C, H, W) format with values in [0, 255].
    Each attack is callable and returns an AttackResult.
    """

    @abstractmethod
    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> 'AttackResult':
        """
        Apply attack to image and optionally transform localization maps.

        The attack applies compatible transformations to the image and localization
        maps (if provided), ensuring the localization maps are valid after the attack.

        Args:
            image: Input image tensor (C, H, W) in [0, 255] range
            localization_map: Optional ground truth localization maps (N_masks, H, W)
                            in [0, 1] range. If provided, will be transformed using
                            the same parameters as the image.

        Returns:
            AttackResult containing:
                - attacked: Transformed image (C, H', W')
                - localization_map: Transformed masks (N_masks, H', W') if input provided,
                                   None otherwise
                - metadata: Attack-specific information
        """
        pass

    @property
    def name(self) -> str:
        """Return the attack class name."""
        return self.__class__.__name__

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable config for this attack instance."""
        def is_simple(value: Any) -> bool:
            return isinstance(value, (str, int, float, bool, type(None)))

        params: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if is_simple(value):
                params[key] = value
                continue
            if isinstance(value, (list, tuple)) and all(is_simple(v) for v in value):
                params[key] = list(value)
                continue
            if isinstance(value, dict):
                if all(isinstance(k, str) and is_simple(v) for k, v in value.items()):
                    params[key] = value
        return params

    @property
    def is_geometry_modifying(self) -> bool:
        """
        Indicates if this attack alters the geometric structure (size, rotation, position).
        Defaults to False (e.g., for Noise, Blur, JPEG).
        """
        return False
