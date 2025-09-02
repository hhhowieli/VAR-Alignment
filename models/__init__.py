from typing import Tuple
import torch.nn as nn

from models.var_model.var import VAR
from models.var_model.vqvae import VQVAE

from models import VAR_GRPO

def build_vae_var_rl(
    device, cfg
) -> Tuple[VQVAE, VAR]:

    var_cfg, vae_cfg = cfg.var, cfg.vqvae
    var_arch = cfg.get("arch", "var")
    patch_nums = vae_cfg.get(
        "patch_nums",
        [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    )

    V = var_cfg.get("V", 4096)
    Cvae = var_cfg.get("Cvae", 32)
    ch = var_cfg.get("ch", 160)
    share_quant_resi = var_cfg.get("share_quant_resi", 4)

    depth = vae_cfg.get("depth", 16)
    attn_l2_norm = vae_cfg.get("attn_l2_norm", True)
    flash_if_available = vae_cfg.get("flash_if_available", True)
    fused_if_available = vae_cfg.get("fused_if_available", True)
    init_adaln = vae_cfg.get("init_adaln", 0.5)
    init_adaln_gamma = vae_cfg.get("init_adaln_gamma", 1e-5)
    init_head = vae_cfg.get("init_head", 0.02)
    init_std = vae_cfg.get("init_std", -1)
    num_classes = vae_cfg.get("num_classes", 1000)
    shared_aln = vae_cfg.get("shared_aln", False)

    if var_arch == "grpo":
        var_cls = VAR_GRPO
    elif var_arch == "var":
        var_cls = VAR
    print(depth)

    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24

    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = var_cls(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)

    return vae_local, var_wo_ddp


from models.reward_models.clip_score import CLIPScoreRewardModel
from models.reward_models.hps_v2 import HPSClipRewardModel
from models.reward_models.pick_score import PickScoreRewardModel

def build_reward_model(
    device, cfg
):
    arch = cfg.get("arch", None)

    if arch == "clipscore":
        clip_score_path = cfg.get("clip_score_path", None)
        return CLIPScoreRewardModel(clip_score_path, device=device)
    elif arch == "hps":
        return HPSClipRewardModel()
    elif arch == "pick":
        return PickScoreRewardModel()
    else:
        raise ValueError(f"Unknown reward model: {arch}")