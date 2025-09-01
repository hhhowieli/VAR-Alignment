import os
import argparse

import torch

import dist

from omegaconf import OmegaConf as OC
from tqdm import tqdm

from trainer import build_trainer
from models import build_vae_var_rl, build_reward_model

def train_one_step(
    args,
    trainer,
    optimizer,
    lr_scheduler,
    loader
):

    optimizer.zero_grad()
    batch = next(loader)

    loss, per_token_loss, per_token_kl = trainer.step(
        batch
    )

    optimizer.backward(loss)
    optimizer.step()
    lr_scheduler.step()

    # total_loss += loss.item()
    # kl_total_loss += per_token_kl.sum(dim=1).mean().item()
    # policy_total_loss += per_token_loss.sum(dim=1).mean().item()

    return loss.item(), per_token_loss.sum(dim=1).mean().item(), per_token_kl.sum(dim=1).mean().item()


def main(args):

    cfg = OC.load(args.config)
    print(cfg)

    (model_cfg, trainer_cfg,
     reward_model_cfg, train_cfg) = cfg.model, cfg.trainer, cfg.reward_model, cfg.train
    var_model = build_vae_var_rl(args.device, model_cfg)
    reward_model = build_reward_model(args.device, reward_model_cfg)

    trainer = build_trainer(
        arch="grpo",
        trainer_cfg=trainer_cfg,
        model=var_model,
        device=args.device,
        reward_model=reward_model
    )

    all_steps = cfg.train.all_steps

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:

        train_one_step(train_cfg, trainer, None, None, None)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    main(args)