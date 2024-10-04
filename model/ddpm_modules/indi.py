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
        out_channel=2,
        lr_reduction=None,
        conditional=True,
        schedule_opt=None,
        val_schedule_opt=None,
        e = 0.01,
    ):
        super().__init__(denoise_fn, image_size, channels=channels, loss_type=loss_type, conditional=conditional, 
                         lr_reduction=lr_reduction,
                         schedule_opt=schedule_opt)
        self.e = e
        self.out_channel = out_channel
        self._t_sampling_mode = 'linear_indi'
        assert self._t_sampling_mode in ['uniform', 'linear_ramp', 'quadratic_ramp', 'linear_indi']
        self._linear_indi_a = 1.0

        self._noise_mode = 'brownian'
        assert self._noise_mode in ['gaussian', 'brownian', 'none']
        if self._noise_mode == 'none':
            self.e = 0.0
        
        self.val_num_timesteps = val_schedule_opt['n_timestep']
        
        msg = f'Sampling mode: {self._t_sampling_mode}, Noise mode: {self._noise_mode}'
        print(f'[{self.__class__.__name__}]: {msg}')

    def set_new_noise_schedule(self, schedule_opt, device):
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
    def p_sample(self, x, t, step_size=None, clip_denoised=True, num_timesteps=None, repeat_noise=False, condition_x=None):
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        if step_size is None:
            step_size = 1.0/ num_timesteps
            
        if t == 0:
            return x
        assert t > 0, "t must be non-negative."

        t_float = torch.Tensor([t/num_timesteps]).to(x.device)
        x0 = self.denoise_fn(x, t_float)
        if clip_denoised:
            x0.clamp_(-1., 1.)

        if self._noise_mode in ['gaussian', 'none']:
            return (step_size/t_float) * x0 + (1 - step_size/t_float) * x
        elif self._noise_mode == 'brownian':
            data_component = (step_size/t_float) * x0 + (1 - step_size/t_float) * x
            if t_float == step_size:
                return data_component
            delta_e = torch.sqrt(self.get_e(t_float-step_size)**2 - self.get_e(t_float)**2)
            noise_component = torch.randn_like(x0) * (t_float - step_size) * delta_e
            
            return data_component + noise_component

    @torch.no_grad()
    def p_sample_loop(self, x_in, clip_denoised=True, continous=False, num_timesteps=None):
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        device = x_in.device
        sample_inter = (1 | (num_timesteps//10))
        assert self.conditional is False
        b = x_in.shape[0]
        
        x_in = torch.cat([x_in]*self.out_channel, dim=1)
        img = x_in + torch.randn_like(x_in)*self.get_t_times_e(torch.Tensor([1.0]).to(device))
        ret_img = img
        t_start = 1
        for i in tqdm(reversed(range(t_start, num_timesteps+1)), desc='sampling loop time step', total=num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=clip_denoised, num_timesteps=num_timesteps)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        
        assert img.shape[0] == 1
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    def get_e(self, t):
        # TODO: for brownian motion, this will change.
        if self._noise_mode in ['gaussian', 'none']:
            return self.e
        elif self._noise_mode == 'brownian':
            assert t > 0, "t must be non-negative."
            return self.e/torch.sqrt(t)
        
    def get_t_times_e(self, t):
        # TODO: for brownian motion, this will change.
        # TODO: the problem is that for brownian motion, we have /sqrt(t). so, it is not defined for t=0.
        # so, this function may be needed. 
        if self._noise_mode in ['gaussian', 'none']:
            return self.e * t
        elif self._noise_mode == 'brownian':
            return self.e * torch.sqrt(t)
    
    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in,clip_denoised=True, continous=False):
        return self.p_sample_loop(x_in, clip_denoised=clip_denoised, continous=continous, num_timesteps=self.val_num_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        raise NotImplementedError("This is not needed.")
    
    def q_sample(self, x_start, x_end, t:float, noise=None):
        assert 0 < t.min(), "t > 0"
        assert t.max() <= 1, "t <= 1. but t is {}".format(t.max())

        if len(t.shape) ==1:
            t = t.reshape(-1, 1, 1, 1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        return (1-t)*x_start + t*x_end + noise * self.get_t_times_e(t)
        
    def sample_t(self, batch_size, device):
        if self._t_sampling_mode == 'linear_ramp':
            # probablity of t=0 is 0, which is what we want.
            probablity =torch.arange(self.num_timesteps)
            probablity = probablity/torch.sum(probablity)
            t = torch.multinomial(probablity,batch_size,replacement=True).to(device).long()
        elif self._t_sampling_mode == 'quadratic_ramp':
            # probablity of t=0 is 0, which is what we want.
            probablity =torch.arange(self.num_timesteps)**2
            probablity = probablity/torch.sum(probablity)
            t = torch.multinomial(probablity,batch_size,replacement=True).to(device).long()
        elif self._t_sampling_mode == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (batch_size,),device=device).long()
        elif self._t_sampling_mode == 'uniform_in_range':
            t = torch.randint((2*self.num_timesteps)//3, self.num_timesteps+1, (batch_size,),device=device).long()
        elif self._t_sampling_mode == 'linear_indi':
            maxv = self.num_timesteps
            t = torch.randint(1, maxv, (batch_size,),device=device).long()
            alpha = 1/(self._linear_indi_a + 1)
            probab = torch.rand(t.shape, device=device)
            mask_for_max = probab > alpha
            t[mask_for_max] = maxv
        return t

    def get_prediction_during_training(self, x_in, noise=None):
        # pass
        x_start = x_in['target']
        x_end = x_in['input']
        # we want to make sure that the shape for x_end is the same as x_start.
        x_end = torch.concat([x_end]*self.out_channel, dim=1)
        b, *_ = x_start.shape
        t = self.sample_t(b, x_start.device)
        t_float = t/self.num_timesteps
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, x_end=x_end, t=t_float, noise=noise)
        assert self.conditional is False
        x_recon = self.denoise_fn(x_noisy, t_float)
        return x_recon

    def p_losses(self, x_in, noise=None):
        x_start = x_in['target']
        x_recon = self.get_prediction_during_training(x_in, noise=noise)    
        loss = self.loss_func(x_start, x_recon)

        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

