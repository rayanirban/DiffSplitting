from model.ddpm_modules.unet import UNet
import torch.nn as nn
import torch

class ForegroundMask(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Conv2d(in_channel,out_channel, 7, padding=3)
    
    def forward(self, x):
        return torch.sigmoid(self.layer(x))
    
class TimePredictor(nn.Module):
    def __init__(self,         in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout=0,
        image_size=128,
        unreal_t_enabled=False,
        ):
        """
        Where unreal_t_enabled=True, we need to predict negative t values and values of t greater than 1. So, relu needs to be removed.
        """
        super().__init__()
        self.unet = UNet(in_channel=in_channel, 
                         out_channel=out_channel, 
                         inner_channel=inner_channel,
                            norm_groups=norm_groups,
                            channel_mults=channel_mults,
                            attn_res=attn_res,
                            res_blocks=res_blocks,
                            dropout=dropout,
                            image_size=image_size,
                         with_time_emb=False)
        self.unreal_t_enabled = unreal_t_enabled
        self.foreground_mask = ForegroundMask(in_channel, out_channel)

    def forward(self, x):
        out = self.unet(x, None)
        if self.unreal_t_enabled is False:
            out = nn.functional.relu(out)
        
        attention = self.foreground_mask(x)
        out = out * attention
        out = out.reshape(out.shape[0], -1)
        return out.sum(dim=1)/ attention.reshape(out.shape).sum(dim=1)

if __name__ == '__main__':
    import torch
    model = TimePredictor()
    x = torch.randn(10, 6, 128, 128)
    out = model(x)
    print(out)