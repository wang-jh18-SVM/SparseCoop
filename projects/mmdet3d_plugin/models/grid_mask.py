import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class Grid(object):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    """
    GridMask: A data augmentation method that applies a grid-like mask to input tensors.

    This augmentation technique creates a grid pattern where some areas are masked out,
    forcing the model to learn from partial information and improving generalization.
    """
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        """
        Args:
            use_h (bool): Whether to apply horizontal grid lines
            use_w (bool): Whether to apply vertical grid lines
            rotate (int): Maximum degrees of rotation for the grid (random from 0 to rotate-1)
            offset (bool): If True, adds random offset values to masked regions
            ratio (float): Controls the width of the grid lines (higher values = wider masked regions)
            mode (int): 0 = masks out grid lines, 1 = keeps grid lines and masks everything else
            prob (float): Probability of applying the grid mask during training
        """
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob  # Starting probability
        self.prob = prob  # Current probability

    def set_prob(self, epoch, max_epoch):
        """Adjusts the probability of applying the mask based on training progress.

        The probability increases linearly with epochs, implementing a form of curriculum learning.

        Args:
            epoch (int): Current training epoch
            max_epoch (int): Maximum number of training epochs
        """
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        """Apply grid mask to input tensor during training with probability self.prob.

        Args:
            x (torch.Tensor): Input tensor of shape (n, c, h, w)

        Returns:
            torch.Tensor: Augmented tensor with grid mask applied, or original tensor
        """
        # Skip mask application randomly based on probability or during evaluation
        if np.random.rand() > self.prob or not self.training:
            return x

        # Get input dimensions
        n, c, h, w = x.size()
        # Reshape to apply same mask to all channels
        x = x.view(-1, h, w)

        # Create larger canvas for the mask (to allow for rotation)
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        # Determine grid size (d) and line width (l)
        d = np.random.randint(2, h)  # Random grid size between 2 and input height
        self.l = min(
            max(int(d * self.ratio + 0.5), 1), d - 1
        )  # Line width proportional to grid size

        # Initialize mask with ones (all pixels kept)
        mask = np.ones((hh, ww), np.float32)

        # Random starting positions for the grid
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        # Create horizontal grid lines (masked regions)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h  # Start position of each horizontal line
                t = min(s + self.l, hh)  # End position (capped at mask height)
                mask[s:t, :] *= 0  # Set grid line region to zero

        # Create vertical grid lines (masked regions)
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w  # Start position of each vertical line
                t = min(s + self.l, ww)  # End position (capped at mask width)
                mask[:, s:t] *= 0  # Set grid line region to zero

        # Randomly rotate the grid mask
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)

        # Crop the mask to the input size from the center of the larger canvas
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        # Convert mask to tensor and move to GPU
        mask = torch.from_numpy(mask.copy()).float().cuda()

        # Invert mask if mode=1 (keep grid lines instead of masking them)
        if self.mode == 1:
            mask = 1 - mask

        # Expand mask to match input dimensions
        mask = mask.expand_as(x)

        # Apply mask to input, with optional random offset for masked regions
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            x = x * mask + offset * (
                1 - mask
            )  # Keep unmasked regions and add offset to masked regions
        else:
            x = x * mask  # Simply zero out masked regions

        # Restore original tensor shape
        return x.view(n, c, h, w)
