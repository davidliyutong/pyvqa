import torch


class Motion(torch.jit.ScriptModule):

    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, frames) -> torch.Tensor:
        """Input could be 4-D or 5-D tensors
        :param ref:
        :param dist:
        :return:
        """
        if len(frames.shape) == 4:
            frames = frames.unsqueeze(0)

        batch_sz = frames.shape[0]
        out = torch.mean(torch.abs(frames[:, 1:].reshape(batch_sz, -1) - frames[:, :-1].reshape(batch_sz, -1)), dim=1)

        return out