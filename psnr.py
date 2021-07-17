"""
File:
    psnr.py
Author:
    厉宇桐
Date:
    2021-01-17
Brief:
    This file defines torch PSNR loss module

"""
import torch


class PSNR(torch.jit.ScriptModule):

    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, ref, dist) -> torch.Tensor:
        """Input could be 4-D or 5-D tensors
        :param ref:
        :param dist:
        :return:
        """
        batch_sz = ref.shape[0]
        diff = torch.sub(ref, dist)
        mse: torch.Tensor = torch.mean(torch.square(diff).reshape(batch_sz, -1), dim=1)
        psnr: torch.Tensor = torch.mul(10, torch.log10(255 ** 2 / mse))
        return torch.clip(psnr, 0, 100)


if __name__ == '__main__':
    print('Test')
    im = torch.randint(0, 255, (5, 3, 256, 256), dtype=torch.float, device='cuda')
    img1: torch.Tensor = torch.div(im, 255)
    img2 = img1
    img2[2, ...] = img2[2, ...] * 0.5
    img2[3, ...] = img2[3, ...] * 0.9

    loss_f = PSNR().cuda()
    loss = loss_f(img1, img2)

    print(loss)
