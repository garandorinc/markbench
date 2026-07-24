import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from typing import Optional

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("dct")
class DCTAttack(AbstractAttack):
    """
    DCT-based attack implementation.

    This attack applies Discrete Cosine Transform based modifications to the image.
    """

    def __init__(self, a=0.25, power=2, argmulti=1.0, setZeros=False, shuffle=False):
        """
        Initialize the DCT attack with specified parameters.

        Args:
            a (float): Weight coefficient for the weight function (default: 0.25)
            power (float): Power for the weight function - any real number (default: 2 for quadratic)
            setZeros (bool): If True, sets all DCT quadrants except first to zero (default: False)
            shuffle (bool): If True, shuffles all DCT coefficients in the last three quadrants (default: False)
        """
        self.a = a
        self.power = power
        self.argmulti = argmulti
        self.setZeros = setZeros
        self.shuffle = shuffle

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

    def plotDCTCoefficients(self, image: torch.Tensor, save_path=None):
        """
        Plot the DCT coefficient maps for R, G, and B channels in one plot.

        Args:
            image: The input BenchmarkImage to analyze
            save_path: Optional path to save the plot
        """
        img_tensor = image.image
        _, height, width = img_tensor.shape

        # Compute DCT for each channel
        dct_channels = []
        channel_names = ["R", "G", "B"]

        for channel in range(min(3, img_tensor.shape[0])):  # Handle up to 3 channels
            channel_np = img_tensor[channel].detach().cpu().numpy()
            dct_channel = torch.from_numpy(
                scipy.fft.dctn(channel_np, type=2, norm="ortho")
            )
            dct_channels.append(dct_channel)

        # Create figure with subplots
        fig, axes = plt.subplots(1, len(dct_channels), figsize=(15, 5))

        if len(dct_channels) == 1:
            axes = [axes]  # Make it iterable for single channel

        for i, (dct_channel, name) in enumerate(zip(dct_channels, channel_names)):
            # Convert to numpy and apply log transform for better visualization
            dct_np = dct_channel.detach().cpu().numpy()

            # Use log transform of absolute values to show both small and large coefficients
            # Add small epsilon to avoid log(0)
            dct_log = np.log1p(np.abs(dct_np))

            # Display the DCT coefficients
            im = axes[i].imshow(dct_log, cmap="viridis", aspect="auto")
            axes[i].set_title(
                f"{name} Channel DCT Coefficients\n(log scale)", fontsize=12
            )
            axes[i].set_xlabel("Frequency (horizontal)")
            axes[i].set_ylabel("Frequency (vertical)")

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label("log(|coeff| + 1)", rotation=270, labelpad=15)

        plt.suptitle(
            f"DCT Coefficient Maps ({image.imageType.upper()})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved DCT coefficient plot to {save_path}")

        plt.show()
        plt.close()

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply the DCT attack and return the attacked tensor.

        Args:
            image: The input tensor to be attacked (C, H, W) in [0, 255]
            localization_map: Optional localization maps (N, H, W) - passed through unchanged

        Returns:
            AttackResult with attacked image and unchanged localization_map
        """
        # Get the image tensor and its dimensions
        img_tensor = image
        _, height, width = img_tensor.shape

        ## Plot original DCT coefficients
        # print("  Plotting original DCT coefficients...")
        # self.plotDCTCoefficients(image)

        # Create the generalized weight map
        weight_map = self.generalizedWeight(width=width, height=height)

        # Apply DCT to each channel and add the weighted random signs
        attacked_tensor = torch.zeros_like(img_tensor)

        for channel in range(img_tensor.shape[0]):
            # Create random +1 or -1 matrix with same size as DCT (50%-50%)
            random_signs = (
                torch.randint(0, 2, (height, width), dtype=torch.float32) * 2 - 1
            )

            # Multiply the random signs with generalized weight elementwise
            weighted_random_signs = random_signs * weight_map

            # Apply true 2D DCT to the channel using scipy
            channel_np = img_tensor[channel].detach().cpu().numpy()
            dct_channel = torch.from_numpy(
                scipy.fft.dctn(channel_np, type=2, norm="ortho")
            )

            # Add the weighted random signs to the DCT
            modified_dct = dct_channel + weighted_random_signs

            # Set all quadrants except first to zero if setZeros is True
            if self.setZeros:
                h_half, w_half = modified_dct.shape[0] // 2, modified_dct.shape[1] // 2
                # Zero out all quadrants except the top-left (first quadrant)
                modified_dct[h_half:, :] = 0  # Bottom quadrants
                modified_dct[:, w_half:] = 0  # Right quadrants
                # Note: The top-left quadrant (rows 0:h_half, cols 0:w_half) remains unchanged

            # Shuffle 63/64 of the coefficients if shuffle is True (keep only top-left 1/64)
            if self.shuffle:
                h_eighth, w_eighth = (
                    modified_dct.shape[0] // 8,
                    modified_dct.shape[1] // 8,
                )

                # Extract coefficients from all 63 quadrants except the top-left (Q1,1)
                quadrant_coeffs = []
                quadrant_positions = []

                # Define 8x8 grid of quadrants
                for row in range(8):
                    for col in range(8):
                        # Skip the top-left quadrant (row=0, col=0) - this is the 1/64 we keep
                        if row == 0 and col == 0:
                            continue

                        # Extract current quadrant
                        start_row, end_row = row * h_eighth, (row + 1) * h_eighth
                        start_col, end_col = col * w_eighth, (col + 1) * w_eighth

                        # Handle edge cases for non-divisible sizes
                        if row == 7:  # Last row
                            end_row = modified_dct.shape[0]
                        if col == 7:  # Last column
                            end_col = modified_dct.shape[1]

                        quadrant_coeffs.append(
                            modified_dct[start_row:end_row, start_col:end_col].flatten()
                        )
                        quadrant_positions.append(
                            (start_row, end_row, start_col, end_col)
                        )

                # Concatenate all coefficients from the 63 quadrants
                all_quadrant_coeffs = torch.cat(quadrant_coeffs)

                # Shuffle them randomly
                shuffled_indices = torch.randperm(all_quadrant_coeffs.size(0))
                shuffled_coeffs = all_quadrant_coeffs[shuffled_indices]

                # Redistribute shuffled coefficients back to the 63 quadrants
                start_idx = 0
                for i, (start_row, end_row, start_col, end_col) in enumerate(
                    quadrant_positions
                ):
                    # Calculate how many coefficients this quadrant needs
                    quadrant_size = (end_row - start_row) * (end_col - start_col)
                    quadrant_shuffled = shuffled_coeffs[
                        start_idx : start_idx + quadrant_size
                    ]

                    # Reshape and place back
                    modified_dct[start_row:end_row, start_col:end_col] = (
                        quadrant_shuffled.reshape(
                            end_row - start_row, end_col - start_col
                        )
                    )
                    start_idx += quadrant_size

                # Note: The top-left 1/64 quadrant (rows 0:h_eighth, cols 0:w_eighth) remains unchanged

            # Apply true 2D inverse DCT to get back to spatial domain
            modified_dct_np = modified_dct.detach().cpu().numpy()
            attacked_channel_np = scipy.fft.idctn(modified_dct_np, type=2, norm="ortho")
            attacked_channel = torch.from_numpy(attacked_channel_np)

            attacked_tensor[channel] = attacked_channel

        # Ensure all values are valid for a picture (clamp to 0-255)
        attacked_tensor = torch.clamp(attacked_tensor, 0, 255)

        # DCT is shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_tensor,
            localization_map=localization_map,
            metadata={}
        )
