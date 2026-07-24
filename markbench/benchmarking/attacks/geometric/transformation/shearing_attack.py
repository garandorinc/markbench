from typing import Optional, Tuple, Union

import torch
import kornia

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult
from markbench.utils.device import get_device


@register_attack("shearing", aliases=["shear"])
class ShearingAttack(AbstractAttack):
    """
    Shearing attack with automatic device selection and canvas expansion.

    Uses Kornia (PyTorch) for both CPU and GPU to ensure pixel-perfect consistency.
    The output canvas is resized to fit the sheared image exactly.

    Args:
        shear_factor_x: Shear factor in X direction.
                       - Float: Exact value (e.g., 0.2 means x' = x + 0.2*y).
                       - Tuple (min, max): Randomly samples from [min, max].
        shear_factor_y: Shear factor in Y direction.
                       - Float: Exact value.
                       - Tuple (min, max): Randomly samples from [min, max].
        device: 'cuda', 'cpu', or '' for auto-detect.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        shear_factor_x: Union[float, Tuple[float, float]] = 0.1,
        shear_factor_y: Union[float, Tuple[float, float]] = 0.1,
        device: str = "",
    ):
        self.shear_factor_x = shear_factor_x
        self.shear_factor_y = shear_factor_y
        self.device = get_device(device)

    def _get_shear_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines the shear factors.
        Calculated on CPU first to ensure seed consistency across devices.
        """
        def resolve(param):
            if isinstance(param, (tuple, list)):
                min_v, max_v = param
                val = torch.rand(1).item()
                return min_v + val * (max_v - min_v)
            return float(param)

        sx = resolve(self.shear_factor_x)
        sy = resolve(self.shear_factor_y)
        
        return (torch.tensor([sx], dtype=torch.float32), 
                torch.tensor([sy], dtype=torch.float32))

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply shearing attack to image and optionally masks.
        """
        img_norm = image.float() / 255.0

        if localization_map is not None:
            to_transform = torch.cat([img_norm, localization_map.float()], dim=0)
        else:
            to_transform = img_norm

        # Add batch dimension: (1, C, H, W)
        to_transform = to_transform.unsqueeze(0)

        if self.device == "cuda" and not to_transform.is_cuda:
            to_transform = to_transform.cuda()
        elif self.device == "cpu" and to_transform.is_cuda:
            to_transform = to_transform.cpu()

        B, C, H, W = to_transform.shape

        sx_tensor, sy_tensor = self._get_shear_params()
        sx = sx_tensor.to(to_transform.device)
        sy = sy_tensor.to(to_transform.device)

        M = torch.zeros((B, 2, 3), dtype=torch.float32, device=to_transform.device)
        M[..., 0, 0] = 1.0
        M[..., 0, 1] = sx
        M[..., 1, 0] = sy
        M[..., 1, 1] = 1.0

        corners = torch.tensor([
            [0.0, 0.0],
            [float(W), 0.0],
            [float(W), float(H)],
            [0.0, float(H)]
        ], device=to_transform.device) # (4, 2)

        M_shear = M[0, :2, :2] # (2, 2)
        
        new_corners = corners @ M_shear.t()
        
        x_coords = new_corners[:, 0]
        y_coords = new_corners[:, 1]

        min_x, max_x = torch.min(x_coords), torch.max(x_coords)
        min_y, max_y = torch.min(y_coords), torch.max(y_coords)

        new_w = int(torch.ceil(max_x - min_x).item())
        new_h = int(torch.ceil(max_y - min_y).item())

        tx = -min_x
        ty = -min_y

        M[..., 0, 2] += tx
        M[..., 1, 2] += ty

        transformed_batch = kornia.geometry.transform.warp_affine(to_transform, M, dsize=(new_h, new_w), padding_mode='zeros')
        transformed_full = transformed_batch.squeeze(0) # (C_total, NewH, NewW)

        if localization_map is not None:
            img_channels = image.shape[0]
            attacked_image = (transformed_full[:img_channels] * 255.0).clamp(0, 255).to(image.device)
            transformed_masks = transformed_full[img_channels:].to(image.device)
            
            transformed_masks = (transformed_masks > 0.5).float()
        else:
            attacked_image = (transformed_full * 255.0).clamp(0, 255).to(image.device)
            transformed_masks = None

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={"shear_x": sx.item(), "shear_y": sy.item()}
        )