import os
import torch
from torch import distributed as dist


class BasicTrainer(object):
    def __init__(self, device: torch.device = torch.device('cuda')):
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        self.world_size = torch.cuda.device_count()
        self.device = device

    def init_distributed(self):
        if self.is_ddp:
            print(f"GPUs are available. The world size is {dist.get_world_size()}."
                  f"\nSetting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                self.device = torch.device(type='cuda', index=0)
            print(f"Using: {self.device}")
