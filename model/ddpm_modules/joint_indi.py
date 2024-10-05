"""
Here, we have two indi models. We also learn a learnable transformation parameter.
"""

from model.ddpm_modules.indi import InDI
import torch.nn as nn
import torch

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
    ):
        super().__init__()
        assert denoise_fn_ch1 is not None, "denoise_fn_ch1 is not provided."
        assert denoise_fn_ch2 is not None, "denoise_fn_ch2 is not provided."
        assert denoise_fn is None, "denoise_fn is not needed."
        self.indi1 = InDI(denoise_fn_ch1, image_size, channels=channels, 
                          loss_type=loss_type, 
                          out_channel = out_channel, 
                          lr_reduction=lr_reduction, 
                          conditional=conditional, 
                          schedule_opt=schedule_opt, 
                          val_schedule_opt=val_schedule_opt, 
                          e=e)

        self.indi2 = InDI(denoise_fn_ch2, image_size, channels=channels, 
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
        print(f'[{self.__class__.__name__}]: w_input_loss: {self.w_input_loss}')
    
    def get_offset(self):
        return self.offset_param
    
    def get_scale(self):
        return self.scale_param
    
    def get_alpha(self):
        return torch.sigmoid(self.alpha_param)
    
    def create_input(self, pred1, pred2):
        """
        It creates the input. The intention is to allow the network to correctly learn the transformation.
        """
        alpha = self.get_alpha()
        offset = self.get_offset()
        scale = self.get_scale()
        correctly_weight_input = alpha * pred1 + (1-alpha) * pred2
        return scale * correctly_weight_input + offset
    

    def p_losses(self, x_in, noise=None):
        x_in_ch1 = {'target': x_in['target'][:,0:1], 'input': x_in['input']}
        x_in_ch2 = {'target': x_in['target'][:,1:2], 'input': x_in['input']}

        x_recon_ch1 = self.indi1.get_prediction_during_training(x_in_ch1, noise=noise)    
        x_recon_ch2 = self.indi2.get_prediction_during_training(x_in_ch2, noise=noise)

        loss_ch1 = self.indi1.loss_func(x_in_ch1['target'], x_recon_ch1)
        loss_ch2 = self.indi2.loss_func(x_in_ch2['target'], x_recon_ch2)
        loss_splitting = (loss_ch1 + loss_ch2) / 2
        pred_input = self.create_input(x_recon_ch1, x_recon_ch2)
        loss_input = self.indi1.loss_func(x_in['input'], pred_input)
        return loss_splitting + self.w_input_loss*loss_input
    
    # @torch.no_grad()
    # def sample(self, batch_size=1, continous=False):
    #     image_size = self.image_size
    #     ch1 = self.indi1.p_sample_loop((batch_size, 1, image_size, image_size), continous)
    #     ch2 = self.indi2.p_sample_loop((batch_size, 1, image_size, image_size), continous)
    #     return torch.cat([ch1, ch2], dim=0)
    
    @torch.no_grad()
    def super_resolution(self, x_in,clip_denoised=True, continous=False):
        ch1 = self.indi1.p_sample_loop(x_in, clip_denoised=clip_denoised, continous=continous, num_timesteps=self.val_num_timesteps)
        ch2 = self.indi2.p_sample_loop(x_in, clip_denoised=clip_denoised, continous=continous, num_timesteps=self.val_num_timesteps)
        return torch.cat([ch1, ch2], dim=0)


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
    