import torch


from omegaconf import OmegaConf as OC



def train_one_step(
    args,
    device,
    trainer,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    timesteps_train, # index
    global_step,
    reward_weights,
):
    total_loss = 0.0
    kl_total_loss = 0.0
    policy_total_loss = 0.0
    total_clip_frac = 0.0
    optimizer.zero_grad()
    (
        caption
    ) = next(loader)

    ref_per_token_logps, per_token_logps, advantages = trainer.GRPO_step(
        args.batch_size,
        label_B = caption
    )

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
def main(args):
    pass

if __name__=="__main__":

    pass