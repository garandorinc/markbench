"""Core interfaces for localized watermark robustness evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Union

from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker


class EmbedStrategy(Enum):
    GLOBAL = "global"
    MASKED = "masked"
    BBOX = "bbox"


class DecodeStrategy(Enum):
    GLOBAL = "global"
    MASKED = "masked"
    BBOX = "bbox"


@jaxtyped(typechecker=typechecker)
@dataclass
class DataPoint:
    """Data structure representing a single image data point or a batch.

    Attributes:
        image: Tensor (C, H, W) or (B, C, H, W)
        masks: Optional tensor of masks
    """

    image: Union[Float[Tensor, "3 height width"], Float[Tensor, "batch 3 height width"]]
    
    masks: Optional[Union[
        Float[Tensor, "n_masks height width"], 
        Float[Tensor, "batch n_masks height width"]
    ]] = None


@jaxtyped(typechecker=typechecker)
@dataclass
class EmbedResult:
    """Result from watermark embedding operation.

    Attributes:
        watermarked: The watermarked image tensor (C, H, W)
        watermark_data: WatermarkData containing embedded message and localization map
        embed_time_ms: Time taken for embedding in milliseconds
        metadata: Additional model-specific metadata
    """

    watermarked: Union[
        Float[Tensor, "3 height width"], 
        Float[Tensor, "batch 3 height width"]
    ]
    watermark_data: 'WatermarkData'
    embed_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@jaxtyped(typechecker=typechecker)
@dataclass
class DecodeResult:
    """Result from watermark decoding operation.

    Attributes:
        watermark_data: WatermarkData containing extracted message and localization map
        decode_time_ms: Time taken for decoding in milliseconds
        metadata: Additional model-specific metadata
    """

    watermark_data: 'WatermarkData'
    decode_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@jaxtyped(typechecker=typechecker)
@dataclass
class WatermarkData:
    """
    Data structure to hold watermark information that the decoder needs to retrieve.

    Attributes:
        message: Message bits (N_masks, N_bits)
        localization_map: Spatial map of watermark presence (N_masks, H, W)
    """

    message: Optional[Union[
        Int[Tensor, "n_masks n_bits"],
        Int[Tensor, "batch n_masks n_bits"]
    ]] = None
    localization_map: Optional[Union[
        Float[Tensor, "n_masks height width"],
        Float[Tensor, "batch n_masks height width"]
    ]] = None


@jaxtyped(typechecker=typechecker)
@dataclass
class AttackResult:
    """Result from attack operation.

    Attributes:
        attacked: The attacked image tensor (C, H', W')
        localization_map: Transformed localization maps (N_masks, H', W') if provided
        attack_time_ms: Time taken for attack in milliseconds
        metadata: Additional attack-specific metadata
    """
    attacked: Union[
        Float[Tensor, "3 height width"], 
        Float[Tensor, "batch 3 height width"]
    ]
    localization_map: Optional[Union[
        Float[Tensor, "n_masks height width"],
        Float[Tensor, "batch n_masks height width"]
    ]] = None
    attack_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    debug_tensors: Dict[str, Tensor] = field(default_factory=dict)

class WatermarkModel(ABC):
    """Abstract base class for watermark models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this model."""
        ...

    @property
    @abstractmethod
    def nbits(self) -> int:
        """Number of bits this model embeds."""
        ...

    @property
    def supports_multiple_messages(self) -> bool:
        """Whether this model supports multiple watermark messages per image.

        Returns:
            True if the model can embed/decode multiple independent messages
            in different regions of the same image, False (default) if it only
            supports one message per image. With False, the runner passes
            exactly one mask and one message per image.
        """
        return False

    @property
    def supported_embed_strategies(self) -> Set[EmbedStrategy]:
        """Embedding strategies this model supports."""
        return {EmbedStrategy.GLOBAL, EmbedStrategy.MASKED}

    @property
    def supported_decode_strategies(self) -> Set[DecodeStrategy]:
        """Decoding strategies this model supports."""
        return {DecodeStrategy.MASKED}

    @abstractmethod
    def embed(
        self,
        image: Float[Tensor, "batch 3 height width"],
        watermark_data: WatermarkData,
    ) -> EmbedResult:
        """Embed watermarks into a batch of images.

        Args:
            image: Input images (B, 3, H, W) with values in [0, 255]
            watermark_data: WatermarkData with message (B, N_masks, N_bits) and
                localization_map (B, N_masks, H, W). Each mask marks the region
                that must carry the corresponding message.

        Returns:
            EmbedResult with the watermarked images (B, 3, H, W) in [0, 255] and
            the WatermarkData that was embedded.
        """
        ...

    @abstractmethod
    def decode(
        self,
        image: Float[Tensor, "batch 3 height width"],
    ) -> DecodeResult:
        """Decode watermarks from a batch of images.

        Args:
            image: Input images (B, 3, H, W) with values in [0, 255]. An attack
                can change H and W.

        Returns:
            DecodeResult whose WatermarkData has message (B, N_decoded, N_bits)
            and localization_map (B, N_decoded, H, W). The localization map is
            required: the decode metrics read it for each image.
        """
        ...
