"""
File:
    iou.py
Author:
    厉宇桐
Date:
    2021-03-29
Brief:
    This file defines torch IOU loss module

"""
import torch


class IOU(torch.jit.ScriptModule):

    def __init__(self, average_size: bool = True):
        super().__init__()
        self.average_size = average_size

    @staticmethod
    @torch.jit.script_method
    def forward(ref, dist) -> torch.Tensor:
        """
        Args:
            ref:  (batch，in_channels，H，W)
            dist: (batch，in_channels，H，W)

        Returns:
            IOU loss

        """
        iou = .0
        batch_num = ref.shape[0]
        for idx in range(batch_num):
            iand1 = torch.sum(ref[idx, :, :, :] * dist[idx, :, :, :])
            ior1 = torch.sum(ref[idx, :, :, :] + torch.sum(dist[idx, :, :, :])) - iand1
            iou1 = iand1 / ior1

            iou += (1 - iou1)

        return iou / batch_num


if __name__ == '__main__':
    print('Test')
    im = torch.randint(0, 255, (5, 3, 256, 256), dtype=torch.float, device='cuda')
    img1 = im / 255
    img2 = img1 * 0.5

    loss_f = IOU().cuda()
    loss = loss_f(img1, img2).mean()

    print(loss.item())
