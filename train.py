import os
import argparse

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import load_dataset

import accelerate
import dist

from omegaconf import OmegaConf as OC
from tqdm import tqdm

from utils.lr_control import filter_params
from trainer import build_trainer
from models import build_vae_var_rl, build_reward_model

def get_dataset():
    def transform_val_examples(examples):
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])
        examples["image"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples

    def transform_train_examples(examples):
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        examples["image"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples

    # @fengsicheng: This way is very slow for big dataset like ImageNet-1K (but can pass the network problem using local dataset)
    # train_set = load_dataset("imagefolder", data_dir=traindir, num_proc=4)
    # test_set = load_dataset("imagefolder", data_dir=valdir, num_proc=4)

    train_set = load_dataset("imagenet-1K", split="train", trust_remote_code=True)                                                                                                                                                                                                            
    test_set = load_dataset("imagenet-1K", split="test", trust_remote_code=True)

    print(train_set["label"])

    train_set.set_transform(transform_train_examples)
    test_set.set_transform(transform_val_examples)

    return train_set, test_set

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
    reward_model.eval()

    names, paras, para_groups = filter_params(var_model, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

    optimizer = torch.optim.AdamW(para_groups, lr=train_cfg.lr)
    
    train_set, test_set = get_dataset()
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    var_model, optimizer, train_loader = accelerate.accelerate(var_model, optimizer, train_loader, device=args.device)

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

        train_one_step(train_cfg, trainer, optimizer, None, train_loader)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    main(args)