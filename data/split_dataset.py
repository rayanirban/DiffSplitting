import numpy as np
import albumentations as A
import os
from skimage.io import imread
from dataclasses import dataclass
from typing import Tuple, Dict, List
# import sys; sys.path.append('..')
from data.cifar10 import load_train_val_data

@dataclass
class DataLocation:
    fpath: str = ''
    channelwise_fpath: Tuple[str]= ()
    directory: str = ''

    def __post_init__(self):
        assert self.fpath or len(self.channelwise_fpath) or self.directory, "At least one of the following must be provided: fpath, channelwise_fpath, directory"
        assert (self.fpath and not self.channelwise_fpath and not self.directory) or (not self.fpath and self.channelwise_fpath and not self.directory) or (not self.fpath and not self.channelwise_fpath and self.directory), "Only one of the following must be provided: fpath, channelwise_fpath, directory"

def load_data(data_type, dataloc:DataLocation)->Dict[int, List[np.ndarray]]:
    if data_type == 'cifar10':
        return load_train_val_data(dataloc.directory, [1,7])
    else:
        if dataloc.fpath:
            return _load_data_fpath(dataloc.fpath)
        elif len(dataloc.channelwise_fpath) > 0:
            return _load_data_channelwise_fpath(dataloc.channelwise_fpath)

def compute_normalization_dict(data_dict, q_val=1.0, uint8_data=False):
    """
    x/x_max [0,1]
    2 x/x_max -1 [-1,1]
    (2x - x_max)/x_max [-1,1]
    (x - x_max/2)/(x_max/2) [-1,1]
    """
    if uint8_data:
        tar_max = 255
        inp_max = 2*tar_max
        img_shape = data_dict[0][0].shape
        if len(img_shape) == 2:
            nC = 1
        else:
            nC = img_shape[0]
        return {
            'mean_input': inp_max/2,
            'std_input': inp_max/2,
            'mean_target': np.array([tar_max/2]*nC*2),
            'std_target': np.array([tar_max/2]*nC*2),
            # 
            'target0_max': tar_max,
            'target1_max': tar_max,
            'input_max': inp_max
        }

    else:
        tar1_max = np.quantile([np.quantile(x, q_val) for x in data_dict[0]], q_val)
        tar2_max = np.quantile([np.quantile(x, q_val) for x in data_dict[1]], q_val)
        inp_max = np.quantile([np.quantile(x+y, q_val) for x,y in zip(data_dict[0],data_dict[1])], q_val)
        return {
            'mean_input': inp_max/2,
            'std_input': inp_max/2,
            'mean_target': np.array([tar1_max/2, tar2_max/2]),
            'std_target': np.array([tar1_max/2, tar2_max/2]),
            # 
            'target0_max': tar1_max,
            'target1_max': tar2_max,
            'input_max': inp_max
        }

def _load_data_channelwise_fpath(fpaths:Tuple[str])-> Dict[int, List[np.ndarray]]:
    assert len(fpaths) == 2, "Only two channelwise fpaths are supported"
    data_ch0 = imread(fpaths[0], plugin='tifffile')
    data_ch1 = imread(fpaths[1], plugin='tifffile')
    return {0: [x for x in data_ch0], 1: [x for x in data_ch1]}

def _load_data_fpath(fpath:str):
    assert fpath.exists(), f"Path {fpath} does not exist"
    assert os.splitext(fpath)[-1] == '.tif', "Only .tif files are supported"
    data = imread(fpath, plugin='tifffile')
    data_ch0 = data[...,0]
    data_ch1 = data[...,1]
    return {0: [x for x in data_ch0], 1: [x for x in data_ch1]}

class SplitDataset:
    def __init__(self, data_type, data_location:DataLocation, patch_size, target_channel_idx = None,random_patching=False, 
                 enable_transforms=False,
                 max_qval=0.98,
                 normalization_dict=None,
                 uncorrelated_channels=False,
                 upper_clip=False):

        assert data_type in ['cifar10','Hagen'], "data_type must be one of ['cifar10','Hagen']"

        self._patch_size = patch_size
        self._data_location = data_location

        # channel_idx is the key. value is list of full sized frames.
        self._data_dict = load_data(data_type, self._data_location)
        self._frameN = min(len(self._data_dict[0]), len(self._data_dict[1]))
        self._target_channel_idx = target_channel_idx
        self._random_patching = random_patching
        self._uncorrelated_channels = uncorrelated_channels

        self._transform = None
        if enable_transforms:
            self._transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5)
                ],
                additional_targets={'image2': 'image'})

        if normalization_dict is None:
            print("Computing mean and std for normalization")
            normalization_dict = compute_normalization_dict(self._data_dict, q_val=max_qval, uint8_data=data_type=='cifar10')

        if upper_clip:
            print("Clipping data to {} quantile".format(max_qval))
            self._data_dict[0] = [np.clip(x, 0, normalization_dict['target0_max']) for x in self._data_dict[0]]
            self._data_dict[1] = [np.clip(x, 0, normalization_dict['target1_max']) for x in self._data_dict[1]]

        assert 'mean_input' in normalization_dict, "mean_input must be provided"
        assert 'std_input' in normalization_dict, "std_input must be provided"
        assert 'mean_target' in normalization_dict, "mean_target must be provided"
        assert 'std_target' in normalization_dict, "std_target must be provided"

        self._mean_inp = normalization_dict['mean_input']
        self._std_inp = normalization_dict['std_input']
        self._mean_target = normalization_dict['mean_target']
        self._std_target = normalization_dict['std_target']
        self._target0_max = normalization_dict['target0_max']
        self._target1_max = normalization_dict['target1_max']
        self._input_max = normalization_dict['input_max']

        assert isinstance(self._mean_target, np.ndarray), "mean_target must be a numpy array"
        assert isinstance(self._std_target, np.ndarray), "std_target must be a numpy array"
        print(self._mean_target.shape)
        # assert len(self._mean_target) == 2, "mean_target must have length 2"
        # assert len(self._std_target) == 2, "std_target must have length 2"
        self._mean_target = self._mean_target.reshape(-1,1,1)
        self._std_target = self._std_target.reshape(-1,1,1)

        msg = f'[{self.__class__.__name__}] Data: {self._frameN}x{len(self._data_dict.keys())}x{self._data_dict[0][0].shape}'
        msg += f' Patch:{patch_size} Random:{int(random_patching)} Aug:{self._transform is not None} Q:{max_qval}'
        if upper_clip is not None:
            msg += f' UpperClip:{int(upper_clip)}'
        msg += f'Uncor:{uncorrelated_channels}'
        print(msg)

    def get_normalization_dict(self):
        assert self._mean_inp is not None, "Mean and std have not been computed"
        
        return {
            'mean_input': self._mean_inp,
            'std_input': self._std_inp,
            'mean_target': self._mean_target,
            'std_target': self._std_target,
            'target0_max': self._target0_max,
            'target1_max': self._target1_max,
            'input_max': self._input_max,
        }
    def normalize_inp(self, inp):
        norm_inp = (inp - self._mean_inp)/self._std_inp
        return norm_inp.astype(np.float32)
    
    def normalize_target(self, target):
        norm_tar = (target - self._mean_target)/self._std_target
        return norm_tar.astype(np.float32)
    
    def patch_count_per_frame(self):
        h,w = self._data_dict[0][0].shape[-2:]
        n_patches_per_frame = (h//self._patch_size) * (w//self._patch_size)
        return n_patches_per_frame
    
    def __len__(self):
        n_patches_per_frame = self.patch_count_per_frame()
        return self._frameN * n_patches_per_frame
    
    def frame_idx(self, index):
        return index // self.patch_count_per_frame()
    
    def patch_loc(self, index):
        frame_idx = self.frame_idx(index)
        index = index % self.patch_count_per_frame()
        h,w = self._data_dict[0][frame_idx].shape[-2:]
        h_idx = index // (h//self._patch_size)
        w_idx = index % (w//self._patch_size)
        return frame_idx, h_idx*self._patch_size, w_idx*self._patch_size


    def __getitem__(self, index):
        if self._random_patching:
            frame_idx = np.random.randint(0, self._frameN)
            h,w = self._data_dict[0][frame_idx].shape[-2:]
            h_idx = np.random.randint(0, h-self._patch_size) if h > self._patch_size else 0
            w_idx = np.random.randint(0, w-self._patch_size) if w > self._patch_size else 0
        else:
            frame_idx, h_idx, w_idx = self.patch_loc(index)
        
        img1 = self._data_dict[0][frame_idx]

        if self._uncorrelated_channels:
            frame_idx = np.random.randint(0, self._frameN)
        img2 = self._data_dict[1][frame_idx]
        
        assert img1.shape == img2.shape, "Images must have the same shape"
        # random h,w location
        patch1 = img1[...,h_idx:h_idx+self._patch_size, w_idx:w_idx+self._patch_size].astype(np.float32)
        patch2 = img2[...,h_idx:h_idx+self._patch_size, w_idx:w_idx+self._patch_size].astype(np.float32)
        if self._transform:
            if patch1.ndim ==3:
                patch1 = patch1.transpose(1,2,0)
                patch2 = patch2.transpose(1,2,0)
            transformed = self._transform(image=patch1, image2=patch2)
            patch1 = transformed['image']
            patch2 = transformed['image2']
            if patch1.ndim ==3:
                patch1 = patch1.transpose(2,0,1)
                patch2 = patch2.transpose(2,0,1)

        inp = patch1 + patch2
        if inp.ndim == 2:
            inp = inp[None]
            patch1 = patch1[None]
            patch2 = patch2[None]
        
        target = np.concatenate([patch1, patch2], axis=0)
        
        inp = self.normalize_inp(inp)
        target = self.normalize_target(target)
        if self._target_channel_idx is None:
            return {'input':inp, 'target':target}
        
        return {'input':inp, 'target':target[self._target_channel_idx: self._target_channel_idx+1]}
    

if __name__ == "__main__":
    import sys
    data_location = DataLocation(channelwise_fpath=('/group/jug/ashesh/data/diffsplit_hagen/val/val_actin-60x-noise2-highsnr.tif',
                                                    '/group/jug/ashesh/data/diffsplit_hagen/val/val_mito-60x-noise2-highsnr.tif'))
    # patch_size = 512
    # data_type = 'hagen'
    # data_location = DataLocation(directory='/group/jug/ashesh/data/cifar-10-python/train')
    patch_size = 256
    data_type = 'Hagen'
    uncorrelated_channels = False
    dataset = SplitDataset(data_type, data_location, patch_size, 
                                max_qval=0.98, upper_clip=True,
                             normalization_dict=None, enable_transforms=True,
                             uncorrelated_channels=True, random_patching=True)
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        inp = data['input']
        target = data['target']
        print(inp.min(), inp.max(),end='\t')
        print(target[0].min(), target[0].max(), end='\t')
        print(target[1].min(), target[1].max())
        # break   


    import matplotlib.pyplot as plt
    data= dataset[0]
    inp = data['input']
    target = data['target']
    _,ax = plt.subplots(figsize=(3,1),ncols=3)
    ax[0].imshow((2+inp.transpose(1,2,0))/4)
    ax[1].imshow((1 +target[:3].transpose(1,2,0))/2)
    ax[2].imshow((1+target[3:].transpose(1,2,0))/2)
    # disable axis
    for a in ax:
        a.axis('off')