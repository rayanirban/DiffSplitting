"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from tifffile import imsave, imread
import xarray as xr
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as LPIPS

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        #if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

        
    return image_numpy

    #return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    #image_pil = Image.fromarray(image_numpy)
    #h, w, _ = image_numpy.shape

    #if aspect_ratio > 1.0:
    #    image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    #if aspect_ratio < 1.0:
    #    image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    #image_pil.save(image_path)
    imsave(image_path, image_numpy)



def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_image_noisy(image, start=0, step=6, number=3):

    mean = [100.416824,100.45542,100.50303,100.54903,100.59541,100.63546,100.67394,100.70801,100.73942,100.76916,100.795105,100.81882,100.841064,100.859795,100.87105,100.87886,100.88255,100.885605,100.88691,100.8873,100.88763]
    std = [1.9103161,1.9412181,1.9788378,2.01439,2.0483522,2.0774746,2.1052907,2.1300852,2.1519976,2.1728525,2.1907341,2.206586,2.2220945,2.234445,2.2421784,2.2465937,2.2488778,2.251113,2.2520072,2.252698,2.251882]

    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image - mean) / std

    return image   

def normalize_image_clean(image, start=0, step=6, number=3):

    mean = [37968.65,74132.914,110701.984,142032.31,174305.97,203156.92,231905.4,258544.11,283130.78,307809.16,329361.44,349770.25,368398.1,384791.7,394938.47,401047.56,404650.56,406744.94,407787.0,408121.6,408121.6]
    std = [119932.95,222743.81,314399.66,384385.72,451080.1,507567.3,561452.1,609623.56,652840.44,695108.44,731161.5,764602.1,794495.06,820216.06,835909.7,845195.56,850561.75,853616.8,855098.3,855563.8,855563.8]
    
    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image - mean) / std

    return image   


def denormalize_image_noisy(image, start=0, step=6, number=3):

    mean = [100.416824,100.45542,100.50303,100.54903,100.59541,100.63546,100.67394,100.70801,100.73942,100.76916,100.795105,100.81882,100.841064,100.859795,100.87105,100.87886,100.88255,100.885605,100.88691,100.8873,100.88763]
    std = [1.9103161,1.9412181,1.9788378,2.01439,2.0483522,2.0774746,2.1052907,2.1300852,2.1519976,2.1728525,2.1907341,2.206586,2.2220945,2.234445,2.2421784,2.2465937,2.2488778,2.251113,2.2520072,2.252698,2.251882]
    
    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image * std) + mean

    return image 

def denormalize_image_clean(image, start=0, step=6, number=3):

    mean = [37968.65,74132.914,110701.984,142032.31,174305.97,203156.92,231905.4,258544.11,283130.78,307809.16,329361.44,349770.25,368398.1,384791.7,394938.47,401047.56,404650.56,406744.94,407787.0,408121.6,408121.6]
    std = [119932.95,222743.81,314399.66,384385.72,451080.1,507567.3,561452.1,609623.56,652840.44,695108.44,731161.5,764602.1,794495.06,820216.06,835909.7,845195.56,850561.75,853616.8,855098.3,855563.8,855563.8]
    
    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image * std) + mean

    return image  
from microsim import schema as ms
import math

def denormalize_haze_clean(image, start=0, step=6, number=3):

    mean = [0.0,36164.27,72733.32,104063.84,136337.17,165188.16,193936.42,220574.77,245161.56,269840.25,291393.0,311801.66,330428.6,346822.9,356969.28,363078.5,366681.8,368775.9,369817.44,370153.5,370153.5]
    std = [0.0,103251.04,195968.27,267290.34,335507.75,393363.1,448593.47,497969.6,542248.25,585539.2,622448.56,656665.5,687243.5,713558.44,729609.6,739106.56,744595.4,747721.8,749238.94,749717.06,749717.06]
    
    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image * std) + mean

    return image  

def denormalize_haze_noisy(image, start=0, step=6, number=3):

    mean = [0.0,0.038292024,0.086004846,0.13222669,0.17840016,0.21855742,0.25680327,0.29103693,0.32219303,0.35220826,0.3782266,0.40173674,0.4238518,0.44275913,0.45400614,0.46170494,0.4654236,0.46853527,0.46985677,0.47018933,0.47058585]
    std = [0.0,2.0244598,2.0346904,2.0486052,2.0655997,2.0820332,2.099508,2.1158442,2.1312904,2.1460204,2.1590917,2.1715577,2.1828532,2.1930034,2.198548,2.2026947,2.204749,2.205849,2.206799,2.2073224,2.2063277]    
    
    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image * std) + mean

    return image

def normalize_haze_noisy(image, start=0, step=6, number=3):

    mean = [0.0,0.038292024,0.086004846,0.13222669,0.17840016,0.21855742,0.25680327,0.29103693,0.32219303,0.35220826,0.3782266,0.40173674,0.4238518,0.44275913,0.45400614,0.46170494,0.4654236,0.46853527,0.46985677,0.47018933,0.47058585]
    std = [0.0,2.0244598,2.0346904,2.0486052,2.0655997,2.0820332,2.099508,2.1158442,2.1312904,2.1460204,2.1590917,2.1715577,2.1828532,2.1930034,2.198548,2.2026947,2.204749,2.205849,2.206799,2.2073224,2.2063277]    

    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image - mean) / std

    return image

def normalize_haze_clean(image, start=0, step=6, number=3):

    mean = [0.0,36164.27,72733.32,104063.84,136337.17,165188.16,193936.42,220574.77,245161.56,269840.25,291393.0,311801.66,330428.6,346822.9,356969.28,363078.5,366681.8,368775.9,369817.44,370153.5,370153.5]
    std = [0.0,103251.04,195968.27,267290.34,335507.75,393363.1,448593.47,497969.6,542248.25,585539.2,622448.56,656665.5,687243.5,713558.44,729609.6,739106.56,744595.4,747721.8,749238.94,749717.06,749717.06]
        
    std = std[start::step]
    mean = mean[start::step]

    std = std[number]
    mean = mean[number]

    image = (image - mean) / std

    return image

sim = ms.Simulation.from_ground_truth(
    ground_truth=np.ones((1, 64, 64), dtype=np.float32),
    scale=(0.04, 0.02, 0.02),
    output_space={"downscale": 1},
    modality=ms.Confocal(pinhole_au=0.5),
    detector=ms.CameraCCD(qe=0.82,full_well=18000,read_noise=6,bit_depth=12,offset=100),
    settings=ms.Settings(random_seed=None, max_psf_radius_aus=4, np_backend="numpy", device='cpu'))



def noising(input_tensor, number, denorm = 'clean'):
    '''
    Add noise to the input tensor
    Parameters:
        input_tensor:  input tensor
    Returns:
        input_tensor: noisy image tensor
    '''
    assert len(input_tensor.shape) == 4, "Input tensor should have 4 dimensions, (batch_size, channels, height, width)"
    Flag1 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor).to(device)
        Flag1 = True
    input_tensor_clone = input_tensor.clone().detach().requires_grad_(False)
    if denorm == 'clean':
        input_tensor_clone = denormalize_image_clean(input_tensor_clone, number=number)
    elif denorm == None:
        input_tensor_clone = input_tensor_clone
    input_tensor_clone = input_tensor_clone.float()
    input_tensor_noise = []
    for i in range (input_tensor_clone.shape[0]):
        inp = input_tensor_clone[i]
        inp = inp.clone().detach().cpu().numpy()   
        inp_noisy = sim.digital_image(inp, with_detector_noise=True, photons_pp_ps_max = 800) # -> why this channel dimension?
        inp_noisy = np.asarray(inp_noisy)
        inp_noisy = inp_noisy[np.newaxis, ...]           
        added_noise = inp_noisy - inp
        input_tensor_noise.append(added_noise)
    input_tensor_noise = np.asarray(input_tensor_noise)
    input_tensor_noise = torch.from_numpy(input_tensor_noise).float().to(device)
    input_tensor_noise.requires_grad = True
    input_tensor_clone = input_tensor_clone + input_tensor_noise
    input_tensor_clone = normalize_image_noisy(input_tensor_clone, number=number)
    noise_added = input_tensor - input_tensor_clone
    input_tensor = input_tensor + noise_added
    if Flag1:
        input_tensor = input_tensor.detach().cpu().numpy()
    return input_tensor 


def lpips(gt, posterior_samples, net_type='squeeze'):
    '''
    Calculate LPIPS score between the ground truth and the prediction
    Parameters:
        gt: ground truth image
        posterior_samples: posterior_samples image
        net_type: network type
    Returns:
        LPIPS score
    '''
    #detach the tensors
    posterior_samples = posterior_samples.detach().cpu()
    gt = torch.tensor(gt).detach().cpu()
    
    mmse_posterior = posterior_samples.mean(axis=1, keepdim=True)
    mmse_posterior = denormalize_image_clean(mmse_posterior, number=0)
    mmse_posterior = 2 * (mmse_posterior - mmse_posterior.min()) / (mmse_posterior.max() - mmse_posterior.min()) - 1    
    mmse_posterior = mmse_posterior.repeat(1, 3, 1, 1)

    gt = denormalize_image_clean(gt, number=0)
    gt = 2 * (gt - gt.min()) / (gt.max() - gt.min()) - 1
    gt = gt.repeat(1, 3, 1, 1)
    
    lpips = LPIPS(gt, mmse_posterior, net_type=net_type).detach().cpu().numpy()

    return lpips.item()