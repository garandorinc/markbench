import random
from typing import List, Optional

import torch
from .abstract_attack import AbstractAttack
from .registry import register_attack
from markbench.interface import AttackResult


@register_attack("chain")
@register_attack("random")
class CompositeAttack(AbstractAttack):
    """Applies multiple attacks sequentially, optionally with random selection."""

    def __init__(
        self,
        attacks: Optional[List[AbstractAttack]] = None,
        count: Optional[int] = None,
        pool: Optional[List[AbstractAttack]] = None,
        replacement: bool = False
    ):
        if attacks is not None:
            self.attacks = attacks
            self.mode = "chain"
            self.count = None
            self.pool = None
        elif count is not None and pool is not None:
            self.count = count
            self.pool = pool
            self.mode = "random"
            self.replacement = replacement
            self.attacks = None
        else:
            raise ValueError("Either attacks or (count, pool) must be provided")

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        if self.mode == "random":
            if self.replacement:
                selected = random.choices(self.pool, k=self.count)
            else:
                selected = random.sample(self.pool, min(self.count, len(self.pool)))
            attacks_to_apply = selected
            combined_metadata = {
                "selected_attacks": [
                    {"name": a.name, **a.get_config()} for a in selected
                ]
            }
        else:
            attacks_to_apply = self.attacks
            combined_metadata = {}

        current_image = image.to(image.device)
        current_map = localization_map.to(image.device)

        is_geometry_modifying_run = any(a.is_geometry_modifying for a in attacks_to_apply)

        for attack in attacks_to_apply:
            result = attack(current_image, current_map)
            current_image = result.attacked.to(image.device)
            current_map = result.localization_map.to(image.device)
            if result.metadata:
                combined_metadata.update(result.metadata)

        combined_metadata["is_geometry_modifying"] = is_geometry_modifying_run

        return AttackResult(
            attacked=current_image,
            localization_map=current_map,
            metadata=combined_metadata
        )

    @property
    def name(self) -> str:
        if self.mode == "random":
            pool_names = ",".join(a.name for a in self.pool)
            return f"random({self.count},{pool_names})"
        else:
            attack_names = ",".join(a.name for a in self.attacks)
            return f"chain({attack_names})"
