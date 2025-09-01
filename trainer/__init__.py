from .trainer_grpo import GRPOTrainer
from .trainer_base import BaseTrainer

def build_trainer(arch, trainer_cfg, **kwargs):
    if arch == "grpo":
        return GRPOTrainer.build_from_cfg(trainer_cfg, **kwargs)