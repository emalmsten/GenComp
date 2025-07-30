from functools import partial
import numpy as np
import torch


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


# Function to get the parameters for q(x_t-1|x_t, x_0)
# So basically parameters to get how to add some noise
def register_schedule(
    beta_schedule="cosine", timesteps=1000,
    linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    betas = make_beta_schedule(
        beta_schedule, timesteps, linear_start=linear_start,
        linear_end=linear_end, cosine_s=cosine_s
    )
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    timesteps, = betas.shape
    num_timesteps = int(timesteps)
    assert alphas_cumprod.shape[0] == num_timesteps, "alphas have to be defined for each timestep"

    to_torch = partial(torch.tensor, dtype=torch.float32)

    register_buffer = {
        "betas": to_torch(betas),
        "alphas_cumprod": to_torch(alphas_cumprod),
        "alphas_cumprod_prev": to_torch(alphas_cumprod_prev),
        # calculations for diffusion q(x_t | x_{t-1}) and others
        "sqrt_alphas_cumprod": to_torch(np.sqrt(alphas_cumprod)),
        "sqrt_one_minus_alphas_cumprod": to_torch(np.sqrt(1. - alphas_cumprod)),
        "log_one_minus_alphas_cumprod": to_torch(np.log(1. - alphas_cumprod)),
        "sqrt_recip_alphas_cumprod": to_torch(np.sqrt(1. / alphas_cumprod)),
        "sqrt_recipm1_alphas_cumprod": to_torch(np.sqrt(1. / alphas_cumprod - 1))
    }

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    v_posterior = 0
    posterior_variance = (
        (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) /
        (1. - alphas_cumprod) + v_posterior * betas
    )
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
    register_buffer.update({
        "posterior_variance": to_torch(posterior_variance),
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        "posterior_log_variance_clipped": to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        "posterior_mean_coef1": to_torch(
        betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)),
        "posterior_mean_coef2": to_torch(
        (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
    })

    # weights for the original loss but not usued bc perform better without it (14) in DDPM paper
    lvlb_weights = (
        betas ** 2 / (2 * posterior_variance * (alphas) *
        (1 - alphas_cumprod))
    )
    lvlb_weights = to_torch(lvlb_weights)
    lvlb_weights[0] = lvlb_weights[1]
    assert not torch.isnan(lvlb_weights).all()
    register_buffer["lvlb_weights"] = lvlb_weights

    return register_buffer
