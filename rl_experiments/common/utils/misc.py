from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random


class LinearDecay:

    def __init__(
        self,
        init_val: float,
        final_val: float,
        total_steps: int
    ):

        self.init_val = init_val
        self.final_val = final_val
        self.total_steps = total_steps

    def __call__(self, cur_step: int):
        if cur_step >= self.total_steps:
            return self.final_val

        return (self.init_val * (self.total_steps - cur_step) +
                self.final_val * cur_step) / self.total_steps
