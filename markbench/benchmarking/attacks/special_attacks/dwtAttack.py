import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from typing import Optional

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("dwt")
class DWTAttack(AbstractAttack):
    """
    DWT-based attack implementation.

    This attack applies Discrete Wavelet Transform based modifications to the image.
    """

    def __init__(self, a=0.25, power=2, argmulti=1.0, wavelet="haar"):
        """
        Initialize the DWT attack with specified parameters.

        Args:
            a (float): Weight coefficient for the weight function (default: 0.25)
            power (float): Power for the weight function - any real number (default: 2 for quadratic)
            argmulti (float): Multiplier for coordinate normalization (default: 1.0)
            wavelet (str): Wavelet type to use (default: 'haar')
        """
        self.a = a
        self.power = power
        self.argmulti = argmulti
        self.wavelet = wavelet

    def generalizedWeight(
        self, a=None, power=None, width=None, height=None, argmulti=None
    ):
        """
        Generate a generalized two-dimensional weight map with specified power.

        The weight function is calculated as:
        xval = -a*(x/width)^power + 1
        yval = -a*(y/height)^power + 1
        returner[x,y] = xval*yval

        Args:
            a (float): Quadratic coefficient (default: self.a)
            power (float): Power for the weight function (default: self.power)
            width (int): Width of the weight map
            height (int): Height of the weight map
            argmulti (float): Multiplier for coordinate normalization (default: self.argmulti)

        Returns:
            torch.Tensor: 2D weight map
        """
        if width is None or height is None:
            raise ValueError("Width and height must be specified")

        # Use instance defaults if not provided
        a = a if a is not None else self.a
        power = power if power is not None else self.power
        argmulti = argmulti if argmulti is not None else self.argmulti

        # Create coordinate tensors
        x = torch.arange(width, dtype=torch.float32)
        y = torch.arange(height, dtype=torch.float32)

        # Calculate x and y values using the generalized formula
        x_norm = argmulti * x / width
        y_norm = argmulti * y / height

        xval = -a * torch.pow(x_norm, power)
        yval = -a * torch.pow(y_norm, power)

        # Create 2D weight map using outer product
        weight_map = torch.outer(yval, xval)
        weight_map = torch.sqrt(torch.abs(weight_map))

        return weight_map

    def plotDWTCoefficients(self, image: torch.Tensor, save_path=None):
        """
        Plot the DWT coefficient maps for R, G, and B channels in one plot.

        Args:
            image: The input BenchmarkImage to analyze
            save_path: Optional path to save the plot
        """
        img_tensor = image.image
        channel_names = ["R", "G", "B"]

        # Create figure with subplots
        fig, axes = plt.subplots(len(channel_names), 4, figsize=(20, 15))

        if len(channel_names) == 1:
            axes = axes.reshape(1, -1)  # Make it 2D for single channel

        for row, channel in enumerate(
            range(min(3, img_tensor.shape[0]))
        ):  # Handle up to 3 channels
            # Convert tensor to numpy for pywt compatibility
            channel_np = img_tensor[channel].detach().cpu().numpy()

            # Apply 2D DWT decomposition
            coeffs = pywt.dwt2(channel_np, self.wavelet, mode="symmetric")

            # Get approximation and detail coefficients
            cA, (cH, cV, cD) = coeffs

            # Plot all 4 subbands
            subband_names = [
                "Approximation (LL)",
                "Horizontal (LH)",
                "Vertical (HL)",
                "Diagonal (HH)",
            ]
            subband_data = [cA, cH, cV, cD]

            for col, (data, name) in enumerate(zip(subband_data, subband_names)):
                # Use log transform for better visualization
                data_log = np.log1p(np.abs(data))

                # Display the DWT coefficients
                im = axes[row, col].imshow(data_log, cmap="viridis", aspect="auto")
                axes[row, col].set_title(
                    f"{channel_names[row]}: {name}\n(log scale)", fontsize=10
                )
                axes[row, col].set_xlabel("Sample index")
                axes[row, col].set_ylabel("Sample index")

                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[row, col])
                cbar.set_label("log(|coeff| + 1)", rotation=270, labelpad=15)

        plt.suptitle(
            f"DWT Coefficient Maps - {self.wavelet.upper()} Wavelet ({image.imageType.upper()})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved DWT coefficient plot to {save_path}")

        plt.show()
        plt.close()

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply the DWT attack and return the attacked tensor.

        Args:
            image: The input tensor to be attacked (C, H, W) in [0, 255]
            localization_map: Optional localization maps (N, H, W) - passed through unchanged

        Returns:
            AttackResult with attacked image and unchanged localization_map
        """
        # Get the image tensor and its dimensions
        img_tensor = image
        _, height, width = img_tensor.shape

        # Apply DWT to each channel and add the weighted random signs
        attacked_tensor = torch.zeros_like(img_tensor)

        for channel in range(img_tensor.shape[0]):
            # Convert tensor to numpy for pywt compatibility
            channel_np = img_tensor[channel].detach().cpu().numpy()

            # Apply 2D DWT decomposition
            coeffs = pywt.dwt2(channel_np, self.wavelet, mode="symmetric")

            # Plot DWT coefficients if requested (only for first channel to avoid duplicates)
            # if self.plotDWTCoefficients and channel == 0:
            #    self.plotDWTCoefficients(image, save_path=f"dwt_coeffs_{image.imageType}.png")
            #
            # Get approximation and detail coefficients
            cA, (cH, cV, cD) = coeffs

            # Create random signs for each coefficient array
            random_signs_cA = torch.randint(0, 2, cA.shape, dtype=torch.float32) * 2 - 1
            random_signs_cH = torch.randint(0, 2, cH.shape, dtype=torch.float32) * 2 - 1
            random_signs_cV = torch.randint(0, 2, cV.shape, dtype=torch.float32) * 2 - 1
            random_signs_cD = torch.randint(0, 2, cD.shape, dtype=torch.float32) * 2 - 1

            # Create weight maps for each coefficient array
            weight_map_cA = self.generalizedWeight(
                width=cA.shape[1], height=cA.shape[0]
            )
            weight_map_cH = self.generalizedWeight(
                width=cH.shape[1], height=cH.shape[0]
            )
            weight_map_cV = self.generalizedWeight(
                width=cV.shape[1], height=cV.shape[0]
            )
            weight_map_cD = self.generalizedWeight(
                width=cD.shape[1], height=cD.shape[0]
            )

            # Multiply random signs with weights
            weighted_cA = torch.from_numpy(cA) + (random_signs_cA * weight_map_cA)
            weighted_cH = torch.from_numpy(cH) + (random_signs_cH * weight_map_cH)
            weighted_cV = torch.from_numpy(cV) + (random_signs_cV * weight_map_cV)
            weighted_cD = torch.from_numpy(cD) + (random_signs_cD * weight_map_cD)

            # Apply inverse DWT
            modified_coeffs = (
                weighted_cA.numpy(),
                (weighted_cH.numpy(), weighted_cV.numpy(), weighted_cD.numpy()),
            )
            attacked_channel_np = pywt.idwt2(
                modified_coeffs, self.wavelet, mode="symmetric"
            )

            # Convert back to tensor
            attacked_channel = torch.from_numpy(attacked_channel_np)

            # Handle size differences due to wavelet transform
            if attacked_channel.shape != img_tensor[channel].shape:
                # Crop or pad to match original size
                orig_h, orig_w = img_tensor[channel].shape
                if attacked_channel.shape[0] > orig_h:
                    attacked_channel = attacked_channel[:orig_h, :orig_w]
                elif attacked_channel.shape[0] < orig_h:
                    attacked_channel = torch.nn.functional.pad(
                        attacked_channel,
                        (
                            0,
                            orig_w - attacked_channel.shape[1],
                            0,
                            orig_h - attacked_channel.shape[0],
                        ),
                    )

                if attacked_channel.shape[1] > orig_w:
                    attacked_channel = attacked_channel[:, :orig_w]
                elif attacked_channel.shape[1] < orig_w:
                    attacked_channel = torch.nn.functional.pad(
                        attacked_channel,
                        (
                            0,
                            orig_w - attacked_channel.shape[1],
                            0,
                            orig_h - attacked_channel.shape[0],
                        ),
                    )

            attacked_tensor[channel] = attacked_channel

        # Ensure all values are valid for a picture (clamp to 0-255)
        attacked_tensor = torch.clamp(attacked_tensor, 0, 255)

        # DWT is shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_tensor,
            localization_map=localization_map,
            metadata={}
        )
