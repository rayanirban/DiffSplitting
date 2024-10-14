import sys; sys.path.append('../')
import pytest
from model.ddpm_modules.joint_indi import JointIndi
import torch

def dummy_denoise_fn(self_, x):
    return x

@pytest.mark.parametrize("n_timestep", [1,2,10])
def test_joint_indi(n_timestep):
    image_size = 512
    val_schedule_opt = {'n_timestep':n_timestep}
    model = JointIndi(
        None,
        image_size,
        channels=1,
        out_channel=1,
        denoise_fn_ch1=dummy_denoise_fn,
        denoise_fn_ch2=dummy_denoise_fn,
        val_schedule_opt=val_schedule_opt,
        conditional=False,)
    
    inp = torch.randn(1,1,image_size,image_size)
    out = model.super_resolution(inp,clip_denoised=True, continous=True)
    assert out.shape[0] == n_timestep + 1
    
