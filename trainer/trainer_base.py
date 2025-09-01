import torch

class BaseTrainer:
    
    def __init__(self):
        pass
    
    def step(self, batch):
        raise NotImplementedError
    
    @classmethod
    def build_from_cfg(cls):
        raise NotImplementedError