import numpy as np
import torch
import torch.nn.functional as F

from gencomp.diffusion.sampling import q_sample


def get_loss(pred, target, loss_type="l2", mean=True):
    pred = pred.float()
    if loss_type == "l1":
        loss = (target - pred).abs()
        if mean:
            loss = loss.mean()
    elif loss_type == "l2":
        if mean:
            loss = F.mse_loss(target, pred)
        else:
            loss = F.mse_loss(target, pred, reduction="none")
    else:
        raise NotImplementedError("unknown loss type '{loss_type}'")
    
    return loss

def p_losses(
    x_start, t, model, register_schedule, parameterization="eps", noise=None,
    is_training=True, config=None, l_simple_weight=1., original_elbo_weight=0., **kwargs
):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(
        x_start=x_start, t=t, noise=noise, config=config,
        sqrt_alphas_cumprod=register_schedule["sqrt_alphas_cumprod"],
        sqrt_one_minus_alphas_cumprod=register_schedule["sqrt_one_minus_alphas_cumprod"],
    )

    model_out = model(x_noisy, t, **kwargs)

    loss_dict = {}
    if parameterization == "eps":
        target = noise
    elif parameterization == "x0":
        target = x_start
    else:
        raise NotImplementedError(f"Parameterization {parameterization} not yet supported")

    loss = get_loss(model_out, target.float(), loss_type="l1", mean=False)
    if loss.dim() == 3:
        loss = loss.unsqueeze(1)
    loss = loss.mean(dim=[1, 2, 3])
    loss = loss.float()
    log_prefix = "train" if is_training else "val"
    
    loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
    loss_simple = loss.mean() * l_simple_weight

    loss_vlb = (register_schedule["lvlb_weights"][t] * loss).mean()
    loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

    loss = loss_simple + original_elbo_weight * loss_vlb
    loss_dict.update({f"{log_prefix}/loss": loss})
    return loss, loss_dict

