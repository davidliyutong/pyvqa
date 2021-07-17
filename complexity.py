import torch
import torch.nn as nn
import torchvision


class DownScale(torch.jit.ScriptModule):
    def __init__(self, input_block: torch.Tensor, num_iteration: int = 4):
        r"""
        Todo: Basically this module need to be completely rewritten
        Args:
            input_block: Normalized tensor (batch, channel, height, width), [0-1]
            num_iteration: Number of maximum iteration
        """
        super().__init__()
        self._input_block = input_block
        self._num_iteration = num_iteration

    @torch.jit.script_method
    def __iter__(self):
        return self

    @torch.jit.script_method
    def __next__(self) -> torch.Tensor:
        if self._num_iteration <= 0:
            raise StopIteration

        res = self._input_block
        _h, _w = res.shape[2:4]

        self._input_block = res[..., ::2, ::2]
        self._num_iteration -= 1
        return res


class DownDynamic(torch.jit.ScriptModule):
    def __init__(self, input_block: torch.Tensor, num_iteration: int = 4):
        r"""
        Todo: Basically this module need to be completely rewritten
        Args:
            input_block:  Normalized tensor (batch, channel, height, width), [0-1]
            num_iteration: Number of maximum iteration
        """
        super().__init__()
        self._input_block = input_block
        self._num_iteration = num_iteration

    @torch.jit.script_method
    def __iter__(self):
        return self

    @torch.jit.script_method

    def __next__(self) -> torch.Tensor:
        if self._num_iteration <= 0:
            raise StopIteration

        res = self._input_block
        ratio = 0.5
        mean = torch.mean(res, dim=(-3, -2, -1), keepdim=True)

        self._input_block = (ratio * res + (1.0 - ratio) * mean).clip(0, 1)
        self._num_iteration -= 1
        return res


class BaseMetric(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    @classmethod
    @torch.jit.script_method
    def _measure(cls, input_block: torch.Tensor) -> torch.Tensor:
        r"""Basic metric:  |I(k,l)− I(k,l −1)| + |I(k,l)− I (k − 1, l)|
        Todo: Basically this module need to be completely rewritten
        Args:
            input_block: Normalized tensor (batch, channel, height, width), [0-1]

        Returns:

        References:
            "A Bit Allocation Method Based on Picture Activity for Still Image Coding",
            Authors: Wook Joong Kim, Jong Won Yi, and Seong Dae Kim


        """
        _batch, _channel_num, _height, _width = input_block.shape

        if _channel_num == 3:
            block_gray: torch.Tensor \
                = 0.299 * input_block[:, 0, :, :] \
                  + 0.587 * input_block[:, 1, :, :] \
                  + .114 * input_block[:, 2, :, :]
            block_gray = block_gray.unsqueeze(1)
        else:
            block_gray = input_block[:, 0, :, :].unsqueeze(1)

        block_gray_shift_left = torch.zeros_like(block_gray)
        block_gray_shift_left[:, :, :, :-1] = block_gray[:, :, :, 1:]
        block_gray_shift_up = torch.zeros_like(block_gray)
        block_gray_shift_up[:, :, :-1, :] = block_gray[:, :, 1:, :]

        diff_vertical = torch.sum(torch.abs((block_gray_shift_up - block_gray))[:, :, :-1]) / (_width * (_height - 1))
        diff_horizontal = torch.sum(torch.abs((block_gray_shift_left - block_gray))[:, :-1, :]) / (_height * (_width - 1))

        return diff_vertical + diff_horizontal

    @torch.jit.script_method
    def forward(self, input_block: torch.Tensor) -> torch.Tensor:
        return self._measure(input_block)


class MultiScaleMetric(BaseMetric):

    def __init__(self):
        super().__init__()

    @classmethod
    @torch.jit.script_method
    def forward(cls, input_block: torch.Tensor) -> torch.Tensor:
        r"""
        Todo: Basically this module need to be completely rewritten
        Args:
            input_block: Normalized tensor (batch, channel, height, width), [0-1]

        Returns:
            A torch Tensor object shape 4 by 4
        """
        _res = list()
        for sample_ in DownScale(input_block, 4):
            for sample__ in DownDynamic(sample_, 4):
                _res.append(cls._measure(sample__))

        res = torch.reshape(torch.Tensor(_res), shape=(4, 4))
        return res


if __name__ == '__main__':
    import numpy as np

    ToImage = torchvision.transforms.ToPILImage()

    test_block = torch.Tensor(np.ones(shape=(1, 3, 16, 16)))

    # # Test iterator
    # for sample in DownScale(test_block, 4):
    #     ToImage(sample[0]).show()
    #
    # for sample in DownDynamic(test_block, 4):
    #     ToImage(sample[0]).show()

    F = MultiScaleMetric()
    F2 = BaseMetric()
    ans = F(test_block)
    ans2 = F2(test_block)
    print(ans)
    print(ans2)
    print("Finished")
