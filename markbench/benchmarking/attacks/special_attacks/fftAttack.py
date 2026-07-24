import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("fft")
class FFTAttack(AbstractAttack):
    """
    FFT-based attack implementation.

    This attack applies Fast Fourier Transform based modifications to the image.
    """

    def __init__(self, a=0.25, power=2, argmulti=1.0, shuffle=False):
        """
        Initialize the FFT attack with specified parameters.

        Args:
            a (float): Weight coefficient for the weight function (default: 0.25)
            power (float): Power for the weight function - any real number (default: 2 for quadratic)
            shuffle (bool): If True, shuffles only the center FFT coefficients (1/4 to 3/4 in x,y) (default: False)
        """
        self.a = a
        self.power = power
        self.argmulti = argmulti
        self.shuffle = shuffle

    def generalizedWeight(
        self, a=None, power=None, width=None, height=None, argmulti=None, device=None
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

        Returns:
            torch.Tensor: 2D weight map
        """
        if width is None or height is None:
            raise ValueError("Width and height must be specified")

        # Use instance defaults if not provided
        a = a if a is not None else self.a
        power = power if power is not None else self.power
        argmulti = argmulti if argmulti is not None else self.argmulti

        # Create coordinate tensors (on the requested device so downstream ops match image device).
        if device is None:
            device = "cpu"
        x = torch.arange(width, dtype=torch.float32, device=device)
        y = torch.arange(height, dtype=torch.float32, device=device)

        # Calculate x and y values using the generalized formula
        x_norm = argmulti * x / width
        y_norm = argmulti * y / height

        xval = -a * torch.pow(x_norm, power)
        yval = -a * torch.pow(y_norm, power)

        # Create 2D weight map using outer product
        weight_map = torch.outer(yval, xval)
        weight_map = torch.sqrt(torch.abs(weight_map))

        return weight_map

    def plotFFTCoefficients(self, image: torch.Tensor, save_path=None):
        """
        Plot the FFT coefficient maps for R, G, and B channels in one plot.

        Args:
            image: The input BenchmarkImage to analyze
            save_path: Optional path to save the plot
        """
        img_tensor = image.image
        _, height, width = img_tensor.shape

        # Compute FFT for each channel
        fft_channels = []
        channel_names = ["R", "G", "B"]

        for channel in range(min(3, img_tensor.shape[0])):  # Handle up to 3 channels
            fft_channel = torch.fft.fft2(img_tensor[channel], norm="ortho")
            fft_channels.append(fft_channel)

        # Create figure with subplots
        fig, axes = plt.subplots(1, len(fft_channels), figsize=(15, 5))

        if len(fft_channels) == 1:
            axes = [axes]  # Make it iterable for single channel

        for i, (fft_channel, name) in enumerate(zip(fft_channels, channel_names)):
            # Convert to numpy and apply log transform for better visualization
            fft_np = fft_channel.detach().cpu().numpy()

            # Use log transform of absolute values to show both small and large coefficients
            # Add small epsilon to avoid log(0)
            fft_log = np.log1p(np.abs(fft_np))

            # Display the FFT coefficients
            im = axes[i].imshow(fft_log, cmap="viridis", aspect="auto")
            axes[i].set_title(
                f"{name} Channel FFT Coefficients\n(log scale)", fontsize=12
            )
            axes[i].set_xlabel("Frequency (horizontal)")
            axes[i].set_ylabel("Frequency (vertical)")

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label("log(|coeff| + 1)", rotation=270, labelpad=15)

        plt.suptitle(
            f"FFT Coefficient Maps ({image.imageType.upper()})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved FFT coefficient plot to {save_path}")

        plt.show()
        plt.close()

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply the FFT attack and return the attacked tensor.

        Args:
            image: The input tensor to be attacked (C, H, W) in [0, 255]
            localization_map: Optional localization maps (N, H, W) - passed through unchanged

        Returns:
            AttackResult with attacked image and unchanged localization_map
        """
        # Get the image tensor and its dimensions
        img_tensor = image
        _, height, width = img_tensor.shape

        ## Plot original FFT coefficients
        # print("  Plotting original FFT coefficients...")
        # self.plotFFTCoefficients(image)

        device = img_tensor.device

        # Create the generalized weight map (match image device so FFT ops don't mix CPU/CUDA).
        weight_map = self.generalizedWeight(width=width, height=height, device=device)

        # Apply FFT to each channel and add the weighted random signs
        attacked_tensor = torch.zeros_like(img_tensor)

        for channel in range(img_tensor.shape[0]):
            # Create random +1 or -1 matrix with same size as FFT (50%-50%)
            random_signs = (
                torch.randint(0, 2, (height, width), dtype=torch.float32, device=device) * 2 - 1
            )

            # Multiply the random signs with generalized weight elementwise
            weighted_random_signs = random_signs * weight_map

            # Apply 2D FFT to the channel
            fft_channel = torch.fft.fft2(img_tensor[channel], norm="ortho")

            # Add the weighted random signs to the FFT
            modified_fft = fft_channel + weighted_random_signs

            # Shuffle center coefficients if shuffle is True (from 1/4 to 3/4 in x and y)
            if self.shuffle:
                # Calculate boundaries for center region (1/4 to 3/4)
                y_start = height // 4
                y_end = 3 * height // 4
                x_start = width // 4
                x_end = 3 * width // 4

                # Extract center coefficients
                center_coeffs = modified_fft[y_start:y_end, x_start:x_end]

                # Flatten, shuffle, and reshape
                center_flat = center_coeffs.flatten()
                shuffled_indices = torch.randperm(center_flat.size(0), device=center_flat.device)
                shuffled_center = center_flat[shuffled_indices]
                modified_fft[y_start:y_end, x_start:x_end] = shuffled_center.reshape(
                    center_coeffs.shape
                )

                # Note: The outer edges (first quarter and last quarter in both directions) remain unchanged

            # Apply inverse FFT to get back to spatial domain
            attacked_channel = torch.fft.ifft2(modified_fft, norm="ortho").real

            attacked_tensor[channel] = attacked_channel

        # Ensure all values are valid for a picture (clamp to 0-255)
        attacked_tensor = torch.clamp(attacked_tensor, 0, 255)

        # FFT is shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_tensor,
            localization_map=localization_map,
            metadata={}
        )
