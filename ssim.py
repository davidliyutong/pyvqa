"""
File:
    ssim_old.py
Author:
    厉宇桐
Date:
    2021-01-17
Brief:
    This file defines torch SSIM and MS-SSIM loss module

References:
    Inspired by https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py

"""

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as torch_f
from typing import Tuple
from tools.torch_filters import gaussian_filter, create_gaussian_kernel_tc


@torch.jit.script
def ssim(ref: torch.Tensor,
         dist: torch.Tensor,
         kernel,
         data_range: float = 1.,
         use_padding: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate ssim index for `x` and `y`
    Args:
        ref: A batch of images
        dist: A batch of images
        kernel: 1-D gaussian kernel
        data_range: Value range of input pixels, 1.0 for normalized image and 255 for 24-bit true color
        use_padding: Bool flag to enable padding

    Returns:
        The ssim value
    """

    k1 = 0.01
    k2 = 0.03
    compensation = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu1 = gaussian_filter(ref, kernel, use_padding)
    mu2 = gaussian_filter(dist, kernel, use_padding)
    mu1_square = mu1.pow(2)  # shape=[5, channel, H - kernel_sz, W - kernel_sz]
    mu2_square = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_square = compensation * (gaussian_filter(ref * ref, kernel, use_padding) - mu1_square)
    sigma2_square = compensation * (gaussian_filter(dist * dist, kernel, use_padding) - mu2_square)
    sigma12 = compensation * (gaussian_filter(ref * dist, kernel, use_padding) - mu1_mu2)

    cs_map = (2 * sigma12 + c2) / (sigma1_square + sigma2_square + c2)  # shape=[5, channel, H-kernel_sz, W-kernel_sz]
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = torch_f.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_square + mu2_square + c1)) * cs_map

    assert isinstance(ssim_map, torch.Tensor)
    ssim_val = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)  # shape=[5, 1]

    return ssim_val, cs


@torch.jit.script
def ms_ssim(ref: torch.Tensor, dist: torch.Tensor, kernel: torch.Tensor, data_range: float, weights,
            use_padding: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate ms-ssim index for `x` and `y`

    Args:
        ref: a batch of images, (batch，in_channels，H，W) reference
        dist: a batch of images, (batch，in_channels，H，W) distorted
        kernel: 1-D gaussian kernel
        data_range: Value range of input pixels, 1.0 for normalized image and 255 for 24-bit true color
        weights: Weights for different levels
        use_padding: Bool flag to enable padding
        eps: Use for avoid NaN gradient.

    Returns:

    Notes:
        The shape and dtype of X and Y must be same
    """

    weights = weights[:, None]  # shape=[5, 1]

    levels = weights.shape[0]  # Default to 5 levels
    values = []
    for i in range(levels):
        ss, cs = ssim(ref, dist, kernel=kernel, data_range=data_range, use_padding=use_padding)

        if i < levels - 1:
            values.append(cs)
            ref = torch_f.avg_pool2d(ref, kernel_size=2, stride=2, ceil_mode=True)
            dist = torch_f.avg_pool2d(dist, kernel_size=2, stride=2, ceil_mode=True)
        else:
            values.append(ss)

    values = torch.stack(values, dim=0)  # shape=[5, Batch, channel]
    # Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
    values = values.clamp_min(eps)

    ms_ssim_val = torch.prod(values ** weights.view(-1, 1, 1), dim=0)

    return ms_ssim_val  # shape=[Batch, channel]


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, kernel_size=11,
                 kernel_sigma=1.5,
                 data_range=255.,
                 channel_num=3,
                 dim_num=4,
                 use_padding=False,
                 channel_averaged=True):
        """
        SSIM loss class
        Args:
            kernel_size: The size of gaussian kernel
            kernel_sigma: The sigma of gaussian normal distribution
            data_range: Value range of input pixels, 1.0 for normalized image and 255 for 24-bit true color
            channel_num: Number of input channels
            dim_num: Number of dimensions, default to 4. like [Batch, C, H, W]
            use_padding: Bool flag to enable padding
            channel_averaged: Bool flag. If true then the value is averaged
        """
        super().__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd.'
        self.channel_num = channel_num
        kernel = create_gaussian_kernel_tc(kernel_size, kernel_sigma)
        kernel = kernel.repeat([channel_num] + [1] * (dim_num - 1))
        kernel = nn.Parameter(kernel)
        self.register_buffer('kernel', kernel)
        self.data_range = data_range
        self.use_padding = use_padding
        self.channel_averaged = channel_averaged

    @torch.jit.script_method
    def forward(self, ref, dist) -> torch.Tensor:
        """

        Args:
            ref: a batch of images, (batch，in_channels，H，W) reference
            dist: a batch of images, (batch，in_channels，H，W) distorted

        Returns:
            SSIM index, shape=(batch,)
        """
        batch_sz = ref.shape[0]
        c, h, w = ref.shape[-3:]

        # Unfold the 2nd dimension of ref
        if len(ref.shape) == 5:
            ref = ref.reshape(-1, c, h, w)
            dist = dist.reshape(-1, c, h, w)

        result, _ = ssim(ref, dist, kernel=self.kernel, data_range=self.data_range, use_padding=self.use_padding)

        # Reshape back
        result = result.reshape(batch_sz, -1, self.channel_num)
        if self.channel_averaged:
            return result.mean(dim=(1, 2))
        else:
            return result.mean(dim=1)


# noinspection PyPep8Naming
class MS_SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding', 'eps']

    def __init__(self, kernel_size=11, kernel_sigma=1.5, data_range=255., channel_num=3, dim_num=4, use_padding=False,
                 weights=None,
                 levels=None,
                 channel_averaged=True,
                 eps=1e-8):
        """
        MS-SSIM loss class
        Args:
            kernel_size: The size of gauss kernel
            kernel_sigma: The sigma of gaussian normal distribution
            data_range: Value range of input pixels, 1.0 for normalized image and 255 for 24-bit true color
            channel_num: Number of input channels
            dim_num: Number of dimensions, default to 4. like [Batch, C, H, W]
            use_padding: Bool flag to enable padding
            channel_averaged: Bool flag. If true then the value is averaged
            weights: Weights for different levels. Default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            levels: Number of downscaling
            eps: Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
        """

        super().__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.channel_averaged = channel_averaged
        self.eps = eps
        self.channel_num = channel_num

        kernel = create_gaussian_kernel_tc(kernel_size, kernel_sigma)
        kernel = kernel.repeat([channel_num] + [1] * (dim_num - 1))
        kernel = nn.Parameter(kernel)
        self.register_buffer('kernel', kernel)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, ref, dist) -> torch.Tensor:
        """

        Args:
            ref: a batch of images, (batch，in_channels，H，W)
            dist: a batch of images, (batch，in_channels，H，W)

        Returns:
            MS-SSIM index, shape=(batch,)
        """
        batch_sz = ref.shape[0]
        c, h, w = ref.shape[-3:]

        # Unfold the 2nd dimension of ref
        if len(ref.shape) == 5:
            ref = ref.reshape(-1, c, h, w)
            dist = dist.reshape(-1, c, h, w)

        result = ms_ssim(ref, dist, kernel=self.kernel, data_range=self.data_range, weights=self.weights,
                         use_padding=self.use_padding, eps=self.eps)

        # Reshape back
        result = result.reshape(batch_sz, -1, self.channel_num)
        if self.channel_averaged:
            return result.mean(dim=(1, 2))
        else:
            return result.mean(dim=1)


if __name__ == '__main__':
    print('Test')
    im = torch.randint(0, 255, (7, 1, 256, 256), dtype=torch.float, device='cuda')
    noise = torch.randint(0, 10, (7, 1, 256, 256), dtype=torch.float, device='cuda')
    img1 = im / 255
    img2 = img1 * 0.5

    loss_f1 = SSIM(data_range=1., channel_num=1, channel_averaged=True).cuda()
    loss1 = loss_f1(img1, img2)
    loss1_ = loss_f1(img1, img1)
    loss1__ = loss_f1(img1, img1 / 1.1)

    loss_f2 = MS_SSIM(data_range=1., channel_num=1).cuda()
    loss2 = loss_f2(img1, img2)
    loss2_ = loss_f2(img1, img1)
    loss2__ = loss_f2(img1, img1 / 1.1)

    print(loss1)
    print(loss1_)
    print(loss1__)

    print(loss2)
    print(loss2_)
    print(loss2__)
