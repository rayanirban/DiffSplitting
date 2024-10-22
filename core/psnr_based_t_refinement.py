"""
Here, the idea is the following:
1. Given an input, use the time classifier to predict the mixing time. 
2. Use the input along with the mixing time to get the channel estimates for the two channels.
3. Use the channel estimates to get a better estimate of the mixing time using Range invariant PSNR as the measure.

If one has multiple samples, one can get the per-sample time and the concensus time. Consensue time is naturally better because it can happen that for some samples,
either the channel estimates are not good or the input is mostly empty. In both cases, per-sample time is often misleading.
"""
from disentangle.core.psnr import RangeInvariantPsnr
import torch
import numpy as np

def get_time_prediction_from_classifier(inp, time_classifier):
    with torch.no_grad():
        pred_t = time_classifier(inp.cuda())
    return pred_t


def get_channel_estimates(inp, indi_1, indi_2, time_classifier):
    # for classifier input = t * c1 + (1-t) * c2
    # For indi_1, time=0 will mean c1 (the target). So, time required for indi_1 will be (1-t), where t is the output of the classifier
    pred_t_2 = get_time_prediction_from_classifier(inp, time_classifier)
    pred_t_1 = 1 - pred_t_2

    pred_ch1 = []
    pred_ch2 = []
    for batch_idx in range(inp.shape[0]):
        ch1 = indi_1.p_sample_loop(inp[batch_idx:batch_idx+1].cuda(), 
                                             clip_denoised=True, continous=False, num_timesteps=1, t_float_start=pred_t_1[batch_idx].item())            
        
        pred_ch1.append(ch1.cpu().numpy())
        ch2 = indi_2.p_sample_loop(inp[batch_idx:batch_idx+1].cuda(), 
                                             clip_denoised=True, continous=False, num_timesteps=1, t_float_start=pred_t_2[batch_idx].item())
        pred_ch2.append(ch2.cpu().numpy())
    
    pred1 = np.concatenate(pred_ch1, axis=0)
    pred2 = np.concatenate(pred_ch2, axis=0)
    return pred1, pred2

def estimate_time_using_PSNR(inp, indi_1, indi_2, time_classifier):
    """
    inp: B,1,H,W : Normalized input
    """
    
    pred1, pred2 = get_channel_estimates(inp, indi_1, indi_2, time_classifier)
    psnr_list = []
    gt = inp.cpu().numpy()[:,0]
    t_list = np.arange(0,1.0,0.05)
    for t in t_list:
        pred = pred1*t + pred2*(1-t)
        psnr_list.append(RangeInvariantPsnr(gt, pred))
    psnr_matrix = torch.stack(psnr_list)
    
    per_sample_t = t_list[psnr_matrix.argmax(dim=0)]
    concensus_t = t_list[psnr_matrix.mean(dim=1).argmax()]
    return per_sample_t, concensus_t
