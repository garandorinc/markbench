from typing import Optional, Tuple, Union

import torch
import kornia

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult
from markbench.utils.device import get_device


@register_attack("rotation", aliases=["rotate"])
class RotationAttack(AbstractAttack):
    """
    Rotation attack with automatic device selection and canvas expansion.
    
    This implementation uses Kornia (PyTorch) for both CPU and GPU to ensure 
    pixel-perfect consistency across devices.

    Args:
        angle: Rotation angle in degrees.
               - If float: Rotates by exactly this amount (signed).
               - If tuple (min, max): Randomly samples from [min, max].
        device: 'cuda', 'cpu', or '' for auto-detect.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self, angle: Union[float, Tuple[float, float]] = 15.0, device: str = ""
    ):
        self.angle = angle
        self.device = get_device(device)

    def _get_rotation_parameters(self) -> torch.Tensor:
        """
        Determines the rotation angle.
        Always calculated on CPU first to ensure seed consistency across devices.
        """
        if isinstance(self.angle, (tuple, list)):
            min_a, max_a = self.angle
            val = torch.rand(1).item() 
            angle_val = min_a + val * (max_a - min_a)
        else:
            angle_val = float(self.angle)
            
        return torch.tensor([angle_val], dtype=torch.float32)

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply rotation attack to image and optionally masks.

        Args:
            image: Input tensor (C, H, W) in [0, 255] range.
            localization_map: Optional ground truth localization maps (N_masks, H, W)

        Returns:
            AttackResult containing rotated (and potentially resized) image/masks.
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
        
        angle_deg = self._get_rotation_parameters().to(to_transform.device)
        
        center = torch.tensor([[W / 2.0, H / 2.0]], device=to_transform.device, dtype=torch.float32)
        scale = torch.ones((1, 2), device=to_transform.device, dtype=torch.float32)
        
        M = kornia.geometry.transform.get_rotation_matrix2d(center, angle_deg, scale)

        angle_rad = torch.deg2rad(angle_deg)
        cos_a = torch.abs(torch.cos(angle_rad))
        sin_a = torch.abs(torch.sin(angle_rad))
        
        new_w = int((H * sin_a) + (W * cos_a))
        new_h = int((H * cos_a) + (W * sin_a))

        tx = (new_w - W) / 2.0
        ty = (new_h - H) / 2.0
        
        M[..., 0, 2] += tx
        M[..., 1, 2] += ty

        transformed_batch = kornia.geometry.transform.warp_affine(to_transform, M, dsize=(new_h, new_w))
        transformed_full = transformed_batch.squeeze(0) # (C_total, NewH, NewW)

        if localization_map is not None:
            img_channels = image.shape[0]
            attacked_image = (transformed_full[:img_channels] * 255.0).clamp(0, 255).to(image.device)
            transformed_masks = (transformed_full[img_channels:] > 0.5).float().to(image.device)
        else:
            attacked_image = (transformed_full * 255.0).clamp(0, 255).to(image.device)
            transformed_masks = None

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={"angle": angle_deg.item()}
        )
