from tqdm import tqdm
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def q_sample(
    x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, config=None
):

    if noise is None:
        noise = torch.randn_like(x_start, device=x_start.device)

    extract_1 = extract_into_tensor(sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)  # shape [batch_size,1,1]
    extract_2 = extract_into_tensor(sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)  # shape [batch_size,1,1]

    X_new = extract_1 * x_start + extract_2 * noise

    use_quantized_noise = False
    if config is not None and hasattr(config, "ltron_params"):
        use_quantized_noise = config.ltron_params.use_quantized_noise
        
    return X_new if not use_quantized_noise else quantize_noise(x_start, X_new, *make_quantize_noise_params(config))


def make_quantize_noise_params(config):
    quantized_params = config.ltron_params.quantized_params

    step_sizes_rot = np.radians(quantized_params.step_sizes_rot) / np.pi
    step_sizes_xyz = np.array(quantized_params.step_sizes_xyz) / config.ltron_params.normalization_factor
    step_sizes = np.concatenate([step_sizes_xyz, step_sizes_rot])
    return step_sizes, quantized_params.noise_mask


def quantize_noise(X, X_new, step_sizes, noise_mask):
    # TODO, there is a small error in the quantized noise where bricks that have been rotated are not longer in a
    #   correctly quantized position for that axis. There is probably a fairly quick fix for this
    if noise_mask is None:
        noise_mask = [1] * X.shape[1]

    noise_mask = torch.tensor(noise_mask, device=X.device).view(1, -1, 1)
    step_sizes = torch.tensor(step_sizes, device=X.device).view(1, -1, 1)

    # The parts under the noise mask will stay the same as X
    X_new = X * (1 - noise_mask) + X_new * noise_mask

    # The translation part is the first 3 elements, the rotation part is the last 3 elements
    X_rot, X_trans = X[:, 3:, :], X[:, :3, :]
    X_new_rot, X_new_trans = X_new[:, 3:, :], X_new[:, :3, :]
    step_sizes_rot, step_sizes_trans = step_sizes[:, 3:, :], step_sizes[:, :3, :]

    # Quantize the rotation part
    X_quant_rot = torch.round((X_new_rot - X_rot) / step_sizes_rot) * step_sizes_rot + X_rot

    N, _, S = X_quant_rot.shape

    # Reshape the rotation part to be (_, 3) for the from euler function
    X_quant_rot_shaped = X_quant_rot.permute(0, 2, 1).reshape(-1, 3)

    # Make the deltas and step sizes in the same format
    deltas = X_new_trans - X_trans
    deltas_shaped = deltas.permute(0, 2, 1).reshape(-1, 3)
    step_sizes_trans_shaped = step_sizes_trans.view(-1, 3)

    # Make the rotation matrices
    rotation = R.from_euler('xyz', X_quant_rot_shaped.cpu() * np.pi)
    rotation_matrix = torch.tensor(rotation.as_matrix(), device=X.device)

    # Rotate the deltas to the reference frame of the quantized rotations (actual brick rotations)
    delta_rotated = torch.einsum('bij,bj->bi', rotation_matrix.float(), deltas_shaped.float())

    # Quantize the translation part
    delta_rotated_quantized = torch.round(delta_rotated / step_sizes_trans_shaped) * step_sizes_trans_shaped

    # Rotate the quantized deltas back to the original frame
    rotation_matrix_inv = torch.tensor(rotation.inv().as_matrix(), device=X.device)
    delta_quantized = torch.einsum('bij,bj->bi', rotation_matrix_inv, delta_rotated_quantized)  # Shape (N, 3)

    # Reshape the quantized deltas back to the original shape
    delta_quantized = delta_quantized.reshape(N, S, 3).permute(0, 2, 1)

    # Add the quantized deltas to the original translation and concatenate with the quantized rotations
    res_trans = X_trans + delta_quantized
    return torch.cat([res_trans, X_quant_rot], dim=1)


# Computes (11) of DDPM paper first part (x_0 basically) related to (7)
def predict_start_from_noise(
    x_t, t, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, noise
):
    return (extract_into_tensor(sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t 
    -  extract_into_tensor(sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * noise)

def q_posterior(x_start, x_t, t, posterior_mean_coef1, posterior_mean_coef2, posterior_variance, posterior_log_variance_clipped):
    posterior_mean = (
        extract_into_tensor(posterior_mean_coef1.to(x_t.device), t, x_t.shape) * x_start +
        extract_into_tensor(posterior_mean_coef2.to(x_t.device), t, x_t.shape) * x_t
    )

    posterior_variance = extract_into_tensor(posterior_variance.to(x_t.device), t, x_t.shape)
    posterior_log_variance_clipped = extract_into_tensor(posterior_log_variance_clipped.to(x_t.device), t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

def p_mean_variance(x, t, model, register_schedule, clip_denoised: bool, **kwargs):
    model_out = model(x, t, **kwargs)
    x_recon = predict_start_from_noise(
        x, t=t, noise=model_out,
        sqrt_recip_alphas_cumprod=register_schedule["sqrt_recip_alphas_cumprod"],
        sqrt_recipm1_alphas_cumprod=register_schedule["sqrt_recipm1_alphas_cumprod"],
    )
    if clip_denoised:
        x_recon.clamp_(-1., 1.)

    # Computes (11) by using (7) again
    model_mean, posterior_variance, posterior_log_variance = q_posterior(
        x_start=x_recon, x_t=x, t=t,
        posterior_mean_coef1=register_schedule["posterior_mean_coef1"],
        posterior_mean_coef2=register_schedule["posterior_mean_coef2"],
        posterior_variance=register_schedule["posterior_variance"],
        posterior_log_variance_clipped=register_schedule["posterior_log_variance_clipped"]
    )
    return model_mean, posterior_variance, posterior_log_variance

# this method returns a t+1 denoised image
@torch.no_grad()
def p_sample(x, t, model, register_schedule, clip_denoised=True, **kwargs):
    b = x.shape[0]

    # apparently better to sample log variance than variance
    model_mean, _, model_log_variance = p_mean_variance(
        x=x, t=t, model=model, register_schedule=register_schedule, clip_denoised=clip_denoised, **kwargs
    )
    noise = torch.randn_like(x)
    # no noise when t == 0
    mask = (1 - (t == 0).float())
    nonzero_mask = mask.reshape(b, *((1,) * (len(x.shape) - 1)))

    # formula from sampling step
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


# predict the noise iteratively and remove it
@torch.no_grad()
def p_sample_loop(
    shape, model, register_schedule, num_timesteps, config = None, x_start = None,
    clip_denoised=True, log_every_t=100, return_intermediates=False, **kwargs
):
    use_quantized_denoise = False
    if config is not None and config.general.dataset == "ltron":
        use_quantized_denoise = config.ltron_params.use_quantized_denoise

    # device = betas.device
    device = model.device
    b = shape[0]

    # image starts as random noise
    img = torch.randn(shape, device=device)

    if use_quantized_denoise:
        # Need to know the original brick positions when using quantized denoise
        t = torch.full((1,), num_timesteps - 1, device=device).long()

        img = q_sample(
            x_start=x_start, t=t, config=config,
            sqrt_alphas_cumprod=register_schedule["sqrt_alphas_cumprod"],
            sqrt_one_minus_alphas_cumprod=register_schedule["sqrt_one_minus_alphas_cumprod"],
            noise=torch.randn(shape, device=device))

    intermediates = [img]
    for i in tqdm(
        reversed(range(0, num_timesteps)),
        desc="Sampling t", total=num_timesteps
    ):
        img = p_sample(
            img,
            torch.full((b,), i, device=device, dtype=torch.long),
            model=model, register_schedule=register_schedule,
            clip_denoised=clip_denoised, **kwargs
        )

        if use_quantized_denoise:
            # Quantize the denoised construction
            img = quantize_noise(x_start, img, *make_quantize_noise_params(config))

        # only save the image every log_every_t steps, and the first and last step
        if return_intermediates and (i % log_every_t == 0 or i == num_timesteps - 1):
            intermediates.append(img)

    if return_intermediates:
        return img, intermediates
    return img