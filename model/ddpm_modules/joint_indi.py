"""
Here, we have two indi models. We also learn a learnable transformation parameter.
"""

from model.ddpm_modules.indi import InDI
import torch.nn as nn
import torch

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
    def super_resolution(self, x_in,clip_denoised=True, continous=False):
        ch1 = self.indi1.p_sample_loop(x_in, clip_denoised=clip_denoised, continous=continous, num_timesteps=self.val_num_timesteps, t_float_start=0.5)
        ch2 = self.indi2.p_sample_loop(x_in, clip_denoised=clip_denoised, continous=continous, num_timesteps=self.val_num_timesteps, t_float_start=0.5)
        if len(x_in.shape) == 4 and len(ch1.shape) == 3:
            ch1 = ch1[None]
            ch2 = ch2[None]
        return torch.cat([ch1, ch2], dim=1)


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
    