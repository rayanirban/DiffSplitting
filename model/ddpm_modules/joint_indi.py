"""
Here, we have two indi models. We also learn a learnable transformation parameter.
"""

from model.ddpm_modules.indi import InDI
import torch.nn as nn
import torch
from core.n2v_utils import update_input_for_n2v
import numpy as np

class IndiCustomT(InDI):
    def sample_t(self, batch_size, device):
        assert self._t_sampling_mode == 'linear_indi'
        
        # this 0.5 will be potentially changing the game. 
        assert self.num_timesteps % 2 == 0, "num_timesteps should be even since we are dividing it by 2 in the next line."
        maxv = int(self.num_timesteps * 0.5)
        t = torch.randint(1, maxv, (batch_size,),device=device).long()
        alpha = 1/(self._linear_indi_a + 1)
        probab = torch.rand(t.shape, device=device)
        mask_for_max = probab > alpha
        t[mask_for_max] = maxv
        return t / self.num_timesteps

class IndiFullTranslation(InDI):
    def sample_t(self, batch_size, device):
        assert self._t_sampling_mode == 'linear_indi'
        
        # this 0.5 will be potentially changing the game. 
        assert self.num_timesteps % 2 == 0, "num_timesteps should be even since we are dividing it by 2 in the next line."
        maxv = int(self.num_timesteps * 0.5)
        t = torch.randint(1, self.num_timesteps, (batch_size,),device=device).long()
        alpha = 1/(self._linear_indi_a + 1)
        probab = torch.rand(t.shape, device=device)
        mask_for_max = probab > alpha
        t[mask_for_max] = maxv
        return t / self.num_timesteps


class JointIndi(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        out_channel=2,
        lr_reduction=None,
        denoise_fn_ch1=None,
        denoise_fn_ch2=None,
        conditional=True,
        schedule_opt=None,
        val_schedule_opt=None,
        w_input_loss = 0.0,
        e = 0.01,
        n2v_p =0.0,
        n2v_kernel_size = 5,
        allow_full_translation=False,
    ):
        super().__init__()
        assert denoise_fn_ch1 is not None, "denoise_fn_ch1 is not provided."
        assert denoise_fn_ch2 is not None, "denoise_fn_ch2 is not provided."
        assert denoise_fn is None, "denoise_fn is not needed."
        indi_class = IndiCustomT if not allow_full_translation else IndiFullTranslation
        self.indi1 = indi_class(denoise_fn_ch1, image_size, channels=channels, 
                          loss_type=loss_type, 
                          out_channel = out_channel, 
                          lr_reduction=lr_reduction, 
                          conditional=conditional, 
                          schedule_opt=schedule_opt, 
                          val_schedule_opt=val_schedule_opt, 
                          e=e)

        self.indi2 = indi_class(denoise_fn_ch2, image_size, channels=channels, 
                          loss_type=loss_type, 
                          out_channel = out_channel, 
                          lr_reduction=lr_reduction, 
                          conditional=conditional, 
                          schedule_opt=schedule_opt, 
                          val_schedule_opt=val_schedule_opt, 
                          e=e)

        self.val_num_timesteps = self.indi1.val_num_timesteps

        self.alpha_param = nn.Parameter(torch.tensor(0.0))
        self.offset_param = nn.Parameter(torch.tensor(0.0))
        self.scale_param = nn.Parameter(torch.tensor(1.0))
        self.w_input_loss = w_input_loss
        self.n2v_p = n2v_p
        self.n2v_kernel_size = n2v_kernel_size
        self.current_log_dict = {}

        print(f'[{self.__class__.__name__}]: w_input_loss: {self.w_input_loss}')
    
    def get_offset(self):
        return self.offset_param
    
    def get_scale(self):
        return self.scale_param
    
    def get_alpha(self):
        return torch.sigmoid(self.alpha_param)
    
    def get_current_log(self):
        return self.current_log_dict
    

    def p_losses(self, x_in, noise=None):
        x_in_ch1 = {'target': x_in['target'][:,0:1], 'input': x_in['target'][:,1:2]}
        x_in_ch2 = {'target': x_in['target'][:,1:2], 'input': x_in['target'][:,0:1]}

        if self.n2v_p > 0.0:
            update_input_for_n2v(x_in_ch1, self.n2v_kernel_size, self.n2v_p)
            update_input_for_n2v(x_in_ch2, self.n2v_kernel_size, self.n2v_p)

        x_recon_ch1 = self.indi1.get_prediction_during_training(x_in_ch1, noise=noise)    
        x_recon_ch2 = self.indi2.get_prediction_during_training(x_in_ch2, noise=noise)

        loss_ch1 = self.indi1.loss_func(x_in_ch1['target'], x_recon_ch1)
        loss_ch2 = self.indi2.loss_func(x_in_ch2['target'], x_recon_ch2)
        loss_splitting = (loss_ch1 + loss_ch2) / 2
        loss_input = 0.0
        
        # self.current_log_dict['loss_input'] = loss_input.item()
        self.current_log_dict['loss_splitting'] = loss_splitting.item()
        self.current_log_dict['alpha'] = self.get_alpha().item()
        self.current_log_dict['offset'] = self.get_offset().item()
        self.current_log_dict['scale'] = self.get_scale().item()
        return loss_splitting + self.w_input_loss*loss_input
    
    # @torch.no_grad()
    # def sample(self, batch_size=1, continous=False):
    #     image_size = self.image_size
    #     ch1 = self.indi1.p_sample_loop((batch_size, 1, image_size, image_size), continous)
    #     ch2 = self.indi2.p_sample_loop((batch_size, 1, image_size, image_size), continous)
    #     return torch.cat([ch1, ch2], dim=0)
    
    @torch.no_grad()
    def predict_one_step(self, ch1_estimate, ch2_estimate, t_cur, for_ch1=False, for_ch2=False, clip_denoised=True):
        """
            When t_cur goes to 0, you get more and more the estimate for which for_ch1 or for_ch2 is True.
        """
        assert for_ch1 or for_ch2, "Either for_ch1 or for_ch2 should be True."
        assert not (for_ch1 and for_ch2), "Either for_ch1 or for_ch2 should be True."
        ch1_new_estimate = ch2_new_estimate = None
        if for_ch1:
            x_in_ch1 = ch1_estimate * (1-t_cur) + ch2_estimate * t_cur
            # TODO: add normalization here. 
            ch1_new_estimate = self.indi1.p_sample_loop(x_in_ch1, clip_denoised=clip_denoised, continous=False, num_timesteps=1, t_float_start=t_cur)
        if for_ch2:
            x_in_ch2 = ch2_estimate * (1-t_cur) + ch1_estimate * t_cur
            # TODO: add normalization here. 
            ch2_new_estimate = self.indi2.p_sample_loop(x_in_ch2, clip_denoised=clip_denoised, continous=False, num_timesteps=1, t_float_start=t_cur)
        return {'ch1': ch1_new_estimate, 'ch2': ch2_new_estimate}

    def _get_t_values(self, t_start, num_timesteps, eps=1e-8):
        t_max = t_start
        step_size = t_max / num_timesteps
        t_min = step_size
        t_values = np.arange(t_max-step_size, t_min-eps, -1*step_size)
        return t_values
    
    @torch.no_grad()
    def predict(self, x_in,clip_denoised=True, continous=False, t_float_start=0.5, num_timesteps=None, eps=1e-8):
        if num_timesteps is None:
            num_timesteps = self.val_num_timesteps

        sample_inter = (1 | (num_timesteps//20))
        ch1_t_start = t_float_start
        ch2_t_start = 1-t_float_start
        ch1_estimate = self.predict_one_step(x_in, x_in, ch1_t_start, for_ch1=True, clip_denoised=clip_denoised)['ch1']
        ch2_estimate = self.predict_one_step(x_in, x_in, ch2_t_start, for_ch2=True, clip_denoised=clip_denoised)['ch2']
        
        ch1_estimate = ch1_estimate[None]
        ch2_estimate = ch2_estimate[None]
        
        ch1_iterative_estimates = ch1_estimate
        ch2_iterative_estimates = ch2_estimate

        if num_timesteps == 1:
            return torch.cat([ch1_iterative_estimates, ch2_iterative_estimates], dim=1)
        
        ch1_t_values = self._get_t_values(ch1_t_start, num_timesteps, eps=eps)
        ch2_t_values = self._get_t_values(ch2_t_start, num_timesteps, eps=eps)
        for idx, tval in enumerate(zip(ch1_t_values, ch2_t_values)):
            ch1_t, ch2_t = tval
            ch1_estimate = self.predict_one_step(ch1_estimate, ch2_estimate, ch1_t, for_ch1=True, clip_denoised=clip_denoised)['ch1']
            ch2_estimate = self.predict_one_step(ch1_estimate, ch2_estimate, ch2_t, for_ch2=True, clip_denoised=clip_denoised)['ch2']
            ch1_estimate = ch1_estimate[None]
            ch2_estimate = ch2_estimate[None]
            if idx % sample_inter == 0:
                ch1_iterative_estimates = torch.cat([ch1_iterative_estimates, ch1_estimate], dim=0)
                ch2_iterative_estimates = torch.cat([ch2_iterative_estimates, ch2_estimate], dim=0)
        
        if continous:
            return torch.cat([ch1_iterative_estimates, ch2_iterative_estimates], dim=1)
        else:
            return torch.cat([ch1_iterative_estimates[-1:], ch2_iterative_estimates[-1:]], dim=1)

        # ch1 = ch2 = None
        # x_in_ch1 = x_in_ch2 = None
        # for t_float in t_values:
        #     if ch1 is None:
        #         x_in_ch1 = x_in
        #         x_in_ch2 = x_in
        #     else:
        #         x_in_ch1 = ch1* (1-t_float) + x_in_ch2 * (1-t_float)
        #         x_in_ch2 = ch2* (1-t_float) + x_in_ch1 * (1-t_float)

        #     ch1 = self.indi1.p_sample_loop(x_in_ch1, clip_denoised=clip_denoised, continous=continous, num_timesteps=1, t_float_start=t_float)
        #     ch2 = self.indi2.p_sample_loop(x_in_ch2, clip_denoised=clip_denoised, continous=continous, num_timesteps=1, t_float_start=t_float)

        # if len(x_in.shape) == 4 and len(ch1.shape) == 3:
        #     ch1 = ch1[None]
        #     ch2 = ch2[None]
        # return torch.cat([ch1, ch2], dim=1)


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    # other functions to make it compatible with the training script
    # they are all taken from model.ddpm_modules.diffusion.GaussianDiffusion
    def set_loss(self, device):
        self.indi1.set_loss(device)
        self.indi2.set_loss(device)
    
    def set_new_noise_schedule(self, schedule_opt, device):
        self.indi1.set_new_noise_schedule(schedule_opt, device)
        self.indi2.set_new_noise_schedule(schedule_opt, device)
    