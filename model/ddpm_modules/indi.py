import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import torch

from model.ddpm_modules.diffusion import GaussianDiffusion, exists, default, make_beta_schedule

class InDI(GaussianDiffusion):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        e = 0.01,
    ):
        super().__init__(denoise_fn, image_size, channels=channels, loss_type=loss_type, conditional=conditional, 
                         schedule_opt=schedule_opt)
        self.e = e

    def set_new_noise_schedule(self, schedule_opt, device):
        # TODO: for brownian motion, this will change.
        self.num_timesteps= schedule_opt['n_timestep']
    

    def q_mean_variance(self, x_start, t):
        raise NotImplementedError("This is not needed.")
    
    def predict_start_from_noise(self, x_t, t, noise):
        raise NotImplementedError("This is not needed.")
    
    def q_posterior(self, x_start, x_t, t):
        raise NotImplementedError("This is not needed.")
    
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        raise NotImplementedError("This is not needed.")

    @torch.no_grad()
    def p_sample(self, x, t, step_size=1, clip_denoised=True, repeat_noise=False, condition_x=None):
        # TODO: for brownian motion, this will change.
        if t == 0:
            return x
        assert t > 0, "t must be non-negative."

        x0 = self.denoise_fn(x, t)
        return (step_size/t) * x0 + (1 - step_size/t) * x

    @torch.no_grad()
    def p_sample_loop(self, x_in, clip_denoised=True, continous=False):
        device = self.x_in.device
        sample_inter = (1 | (self.num_timesteps//10))
        assert self.conditional is False
        shape = x_in
        b = shape[0]
        img = torch.randn(shape, device=device)
        ret_img = img
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=clip_denoised)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        return img

    def get_e(self, t):
        # TODO: for brownian motion, this will change.
        return self.e

    def get_t_times_e(self, t):
        # TODO: for brownian motion, this will change.
        # TODO: the problem is that for brownian motion, we have /sqrt(t). so, it is not defined for t=0.
        # so, this function may be needed. 
        return self.e * t
    
    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in,clip_denoised=True, continous=False):
        return self.p_sample_loop(x_in, clip_denoised=clip_denoised, continous=continous)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        raise NotImplementedError("This is not needed.")
    
    def q_sample(self, x_start, x_end, t, noise=None):
        if len(t.shape) ==1:
            t = t.reshape(-1, 1, 1, 1)
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (1-t)*x_start + t*x_end + noise * self.get_t_times_e(t)

    def p_losses(self, x_in, noise=None):
        # pass
        x_start = x_in['target']
        x_end = x_in['input']
        # we want to make sure that the shape for x_end is the same as x_start.
        x_end = torch.concat([x_end, x_end], dim=1)

        b, *_ = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, x_end=x_end, t=t, noise=noise)
        assert self.conditional is False
        x_recon = self.denoise_fn(x_noisy, t)
        loss = self.loss_func(x_start, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
