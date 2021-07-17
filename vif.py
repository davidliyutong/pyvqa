import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from typing import List
from tools.cv2_filters import create_gaussian_kernel_nd

"""
-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2005 The University of Texas at Austin
All rights reserved.
Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
at the University of Texas at Austin (UT Austin, 
http://www.utexas.edu), is acknowledged in any publication that reports research using this code. The research
is to be cited in the bibliography as:
H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality", IEEE Transactions on 
Image Processing, (to appear).
IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------

This is a pytorch implementation of VIF metrics
"""


class VIF(torch.jit.ScriptModule):
    esp: float = 1e-10

    def __init__(self, sigma_nsq: int = 2, eps=1e-10, channel_num=3, dim_num=4, size_averaged: bool = True):
        r"""
        Args:
            size_averaged: If the output is averaged
            sigma_nsq:  The sigma value (default 2)
            eps: The epsilon value (default 1e-10)
            channel_num: Number of channels
            dim_num: Number of dimensions. Defaut to 4 [B, C, H, W]
        """
        super().__init__()
        self.sigma_nsq = sigma_nsq
        self.eps = eps
        self.sigmas: List[float] = [(2 ** (4 - scale + 1) + 1) / 5.0 for scale in range(1, 5)]
        self.channel_num = channel_num
        self.size_averaged = size_averaged
        self.dim_num = dim_num

        # "Indeed, the function gaussian_filter is implemented by
        #  applying multiples 1D gaussian filters (you can see
        #  that here). This function uses gaussian_filter1d which
        #  generate itself the kernel using _gaussian_kernel1d
        #  with a radius of int(truncate * sigma + 0.5)."
        #
        # -> https://stackoverflow.com/questions/60798600/why-scipy- \
        #    ndimage-gaussian-filter-doesnt-have-a-kernel-size
        #
        # "Truncate the filter at this many standard deviations.
        #  Default is 4.0."
        #
        # -> https://docs.scipy.org/doc/scipy/reference/generated/ \
        #    scipy.ndimage.gaussian_filter.html
        self.kernel_sizes = [int(4.0 * sd + 0.5) for sd in self.sigmas]

        kernels = [nn.Parameter(torch.tensor(create_gaussian_kernel_nd(2, 4, kernel_size=self.kernel_sizes[i], sigma=self.sigmas[i]), requires_grad=False).repeat(channel_num, 1, 1, 1)) for i in range(len(self.sigmas))]

        # Warning: ugly workaround for DataParallel, use register_buffer to prevent them from modified
        self.register_buffer('kernel0', kernels[0])
        self.register_buffer('kernel1', kernels[1])
        self.register_buffer('kernel2', kernels[2])
        self.register_buffer('kernel3', kernels[3])

        # kernels0： 4 kernels shape=[channel_num, 1, 1, kernel_sz]
        # channel_num is usually 3

    @torch.jit.script_method
    def _filter(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        device = x.device
        if idx == 0:
            return torch_f.conv2d(x, self.kernel0.to(device), groups=self.channel_num)
        if idx == 1:
            return torch_f.conv2d(x, self.kernel1.to(device), groups=self.channel_num)
        if idx == 2:
            return torch_f.conv2d(x, self.kernel2.to(device), groups=self.channel_num)
        if idx == 3:
            return torch_f.conv2d(x, self.kernel3.to(device), groups=self.channel_num)
        else:
            raise ValueError('idx is not legal')

    def _gen_pyramid(self, mat: torch.Tensor) -> List[torch.Tensor]:
        out: List[torch.Tensor] = [mat]
        for idx in range(1, 4):
            out.append(self._filter(out[-1], idx)[:, :, ::2, ::2])
        return out

    @torch.jit.script_method
    def forward(self, ref: torch.Tensor, dist: torch.Tensor):
        r"""Visual Information Fidelity measure on x, y, ranging from 0-255
        Args:
            ref: reference image, [0,1] 4-D or 5-D tensor
            dist: distorted image, [0,1] 4-D or 5-D tensor

        Returns:
            The VIF value

        """
        # Remember shape
        device = ref.device
        batch_sz = ref.shape[0]
        c, h, w = ref.shape[-3:]

        # Unfold the 2nd dimension of ref
        if len(ref.shape) == 5:
            ref = ref.reshape(-1, c, h, w)
            dist = dist.reshape(-1, c, h, w)

        # Create placeholder
        _batch_sz_tmp = ref.shape[0]
        num = torch.zeros(size=(_batch_sz_tmp, 4)).to(device)  # Place holder
        den = torch.zeros(size=(_batch_sz_tmp, 4)).to(device)  # Place holder

        ref_pyramid: List[torch.Tensor] = self._gen_pyramid(ref)
        dist_pyramid: List[torch.Tensor] = self._gen_pyramid(dist)

        for idx, scale in enumerate(range(1, 5)):
            mu1 = self._filter(ref_pyramid[idx], idx)
            mu2 = self._filter(dist_pyramid[idx], idx)
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = self._filter(ref_pyramid[idx] * ref_pyramid[idx], idx) - mu1_sq
            sigma2_sq = self._filter(dist_pyramid[idx] * dist_pyramid[idx], idx) - mu2_sq
            sigma12 = self._filter(ref_pyramid[idx] * dist_pyramid[idx], idx) - mu1_mu2

            sigma1_sq = sigma1_sq.clip(0, float('inf'))
            sigma2_sq = sigma2_sq.clip(0, float('inf'))

            g = sigma12 / (sigma1_sq + self.eps)
            sv_sq = sigma2_sq - g * sigma12

            mask1 = sigma1_sq < self.eps
            mask2 = sigma2_sq < self.eps

            g = g - g * mask1
            sv_sq = sv_sq - sv_sq * mask1 + sigma2_sq * mask1
            sigma1_sq = sigma1_sq.clip(self.eps, 1)

            g = g - g * mask2
            sv_sq = sv_sq * mask2

            mask3 = g < 0
            sv_sq = sv_sq - sv_sq * mask3 + sigma2_sq * mask3

            g = g.clip(0, 1)
            sv_sq = sv_sq.clip(self.eps, 1)

            num[:, idx] = torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + self.sigma_nsq)), dim=(1, 2, 3))
            den[:, idx] = torch.sum(torch.log10(1 + sigma1_sq / self.sigma_nsq), dim=(1, 2, 3))

        # Reshape back
        num = num.reshape(batch_sz, -1, 4)
        den = den.reshape(batch_sz, -1, 4)
        if self.size_averaged:
            return torch.sum(num, dim=(1, 2)) / torch.sum(den, dim=(1, 2))
        else:
            return torch.sum(num, dim=1) / torch.sum(den, dim=1)


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
    torch.cuda.set_device(1)

    print('Test')
    DEVICE = torch.device(0)

    im_0 = torch.randint(0, 255, (5, 3, 25, 25), dtype=torch.float).to(DEVICE)
    img1 = im_0 / 255
    img2 = img1 * 0.5

    loss_f_0 = VIF(channel_num=3).to(DEVICE)

    loss_0 = loss_f_0(img1, img2)
    loss_1 = loss_f_0(img1, img1)
    loss_2 = loss_f_0(img1, img1 / 1.1)
    print(loss_0)
    print(loss_1)
    print(loss_2)
    print('------------')

    im_1 = torch.randint(0, 255, (2, 5, 3, 25, 25), dtype=torch.float).to(DEVICE)
    img3 = im_1 / 255
    img4 = img3 * 0.5
    loss_4 = loss_f_0(img3, img4)
    loss_5 = loss_f_0(img3, img3)
    loss_6 = loss_f_0(img3, img3 / 1.1)
    print(loss_4)
    print(loss_5)
    print(loss_6)
    print('------------')

    loss_f_1 = VIF(channel_num=3, size_averaged=False).to(DEVICE)

    loss_7 = loss_f_1(img3, img4)
    loss_8 = loss_f_1(img3, img3)
    loss_9 = loss_f_1(img3, img3 / 1.1)
    print(loss_7)
    print(loss_8)
    print(loss_9)
    print('------------')
