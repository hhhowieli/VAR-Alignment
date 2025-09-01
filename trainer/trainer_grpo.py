from functools import partial
from typing import Optional, Tuple, Union

import torch

from .trainer_base import BaseTrainer

class GRPOTrainer(BaseTrainer):
    def __init__(self, model, device, reward_model, cfg):

        self.var = model

        self.RM = reward_model

        self.cfg = cfg
        self.beta = cfg.beta
        self.clip_range = cfg.clip_range

        self.device = device

    def get_advantages(self, samples):
        images = samples["images"]
        text = samples["label_B"]
        images = images.mul_(255).cpu()

        if self.cfg.rm_arch == "hpsv2":
            rewards = self.RM(images, text)
        elif self.cfg.rm_arch == "clipscore":
            rewards = self.RM(images, text)

        samples["reward"] = rewards

        advantages = (rewards - rewards.mean())/(rewards.std()+1e-8)
        samples["advantages"] = advantages

        return advantages


    def get_per_token_logps(self, logits_BlVs, idx_Bls):
        per_token_logps = [] # Use a loop to reduce memory peak.
        for logits_row, input_ids_row in zip(logits_BlVs, idx_Bls):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def GRPO_step(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
    ):
        g_seed = g_seed or self.cfg.seed
        cfg = cfg or self.cfg.cfg
        top_k = top_k or self.cfg.top_k
        top_p = top_p or self.cfg.top_p
        more_smooth = more_smooth or self.cfg.more_smooth

        samples = self.var.sample_reference_model(
            B, label_B, g_seed, cfg, top_k, top_p, more_smooth
        )

        logits_BlVs_ref = samples["logits_BlVs_ref"]
        logits_BlVs = samples["logits_BlVs"]
        idx_Bls = samples["idx_Bls"]

        advantages = self.get_advantages(samples)

        ref_per_token_logps = self.get_per_token_logps(logits_BlVs_ref, idx_Bls)
        per_token_logps = self.get_per_token_logps(logits_BlVs, idx_Bls)

        ratio = torch.exp(per_token_logps - ref_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        loss = per_token_loss + self.beta * per_token_kl
        loss = -loss.sum(dim=1).mean()

        return loss, per_token_loss, per_token_kl

    def step(self, batch):
        pass
    
    @classmethod
    def build_from_cfg(cls, cfg, device, **kwargs):

        model = kwargs.get("model", None)
        reward_model = kwargs.get("reward_model", None)
        
        return cls(model, device, reward_model, cfg)

if __name__ == "__main__":

    trainer = GRPOTrainer(None)

    x = torch.randn(1, 10, 10)
    y = torch.randint(0, 10, (1, 10))

    print(x, y)
    print(trainer.get_per_token_logps(x, y))