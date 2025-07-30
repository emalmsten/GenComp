from omegaconf import OmegaConf
import lightning.pytorch as L
import torch

from gencomp.diffusion.loss import p_losses
from gencomp.diffusion.sampling import p_sample_loop, q_sample
from gencomp.diffusion.scheduling import register_schedule
from gencomp.util import count_params, instantiate_from_config

class DDPM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = OmegaConf.load(config)
        self.num_timesteps = self.config.model.params.timesteps
        self.register_schedule = register_schedule(timesteps=self.num_timesteps, beta_schedule='linear', linear_start=1e-4, linear_end=2e-3)

        self.model = DiffusionWrapper(self.config.model.params.model, config=self.config)
        count_params(self.model, verbose=True)


    def visualize_noise(self, x, steps = 50, config=None):
        """Method for returning the noise at each timestep"""
        noise_ar = []
        b = 1

        noise = torch.randn_like(x, device=x.device)
        for i in range(self.num_timesteps):
            t = torch.full((b,), i, device=x.device, dtype=torch.long)

            qs = q_sample(x, t, self.register_schedule["sqrt_alphas_cumprod"],
                          self.register_schedule["sqrt_one_minus_alphas_cumprod"], noise=noise, config=config)

            # Only append the noise every 'steps' and the first and last timestep
            if i % steps == 0 or i == 0 or i == self.num_timesteps - 1:
                noise_ar.append(qs)

        return noise_ar


    def forward(self, x, *args, **kwargs):
        # x[0] should be the data
        # x[1] should be the conditionals otherwise empty Dict

        t = torch.randint(0, self.num_timesteps, (x[0].shape[0],), device=self.device).long()

        self.register_schedule = {key: tensor.to(self.device) for key, tensor in self.register_schedule.items()}

        return p_losses(x[0].float(), t, self.model, self.register_schedule,
                        config = self.config, *args, **x[1], **kwargs)

    def shared_step(self, batch):
        # x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(batch)
        return loss, loss_dict
        
    # def training_step(self, batch, batch_idx):
    #     inputs, target = batch
    #     output = self(inputs, target)
    #     loss = torch.nn.functional.nll_loss(output, target.view(-1))
    #     return loss

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.base_learning_rate
        )

    @torch.no_grad()
    def sample(self, batch_size=16, config = None, return_intermediates=False, log_every_t=100, x_start = None, **kwargs):

        return p_sample_loop(
            (batch_size, *self.config.model.params.sample_shape),
            model=self.model,
            register_schedule=self.register_schedule,
            num_timesteps=self.num_timesteps,
            return_intermediates=return_intermediates,
            log_every_t=log_every_t,
            config=config,
            x_start=x_start,
            **kwargs
        )


class DiffusionWrapper(L.LightningModule):
    def __init__(self, diff_model_config, config=None):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config, whole_config=config)


    def forward(self, x, t, **kwargs):
        return self.diffusion_model(x, t, **kwargs)