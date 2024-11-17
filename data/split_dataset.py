import numpy as np
# import albumentations as A
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

def compute_normalization_dict(data_dict, channel_weights:List[float], q_val=1.0, uint8_data=False):
    """
    x/x_max [0,1]
    2 x/x_max -1 [-1,1]
    (2x - x_max)/x_max [-1,1]
    (x - x_max/2)/(x_max/2) [-1,1]
    """
    if uint8_data:
        tar_max = 255
        inp_max = tar_max * np.sum(channel_weights)
        img_shape = data_dict[0][0].shape
        tar1_max = tar_max
        tar2_max = tar_max
        if len(img_shape) == 2:
            nC = 1
        else:
            nC = img_shape[0]
        return {
            'mean_input': inp_max/2,
            'std_input': inp_max/2,
            'mean_target': np.array([tar1_max/2]*nC + [tar2_max/2]*nC),
            'std_target': np.array([tar1_max/2]*nC + [tar2_max/2]*nC),
            # 
            'target0_max': tar1_max,
            'target1_max': tar2_max,
            'input_max': inp_max
        }

    else:
        tar1_unravel = np.concatenate([x.reshape(-1,) for x in data_dict[0]])
        tar2_unravel = np.concatenate([x.reshape(-1,) for x in data_dict[1]])
        tar1_max = np.quantile(tar1_unravel, q_val)
        tar2_max = np.quantile(tar2_unravel, q_val)
        inp_max = np.quantile(tar1_unravel*channel_weights[0]+(
                              tar2_unravel*channel_weights[1]), 
                              q_val)
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
    print('HARDCODED upperclip to 1993. Disable it if not needed !!!')
    data_ch0[data_ch0 > 1993.0] = 1993.0
    data_ch1[data_ch1 > 1993.0] = 1993.0
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
                 channel_weights=None,
                 input_from_normalized_target=False,
                 upper_clip=False):
        """
        Args:
        data_type: str - 'cifar10' or 'Hagen'
        data_location: DataLocation - location of the data (file path or directory)
        patch_size: int - size of the patch on which the model will be trained
        target_channel_idx: int - While the input is created from both channels, this decides which target needs to be predicted. If None, both channels are used as target.
        random_patching: bool - If True, random patching is done. Else, patches are extracted in a grid.
        enable_transforms: bool - If True, data augmentation is enabled.
        max_qval: float - quantile value for clipping the data and for computing the max value for the normalization dict.
        normalization_dict: dict - If provided, the normalization dict is used. Else, it is computed.
        uncorrelated_channels: bool - If True, the two diffrent random locations are used to crop patches from the two channels. Else, the same location is used.
        channel_weights: list - Input is the weighted sum of the two channels. If None, the weights are set to 1.
        upper_clip: bool - If True, the data is clipped to the max_qval quantile value.
        """

        assert data_type in ['cifar10','Hagen'], "data_type must be one of ['cifar10','Hagen']"

        self._patch_size = patch_size
        self._data_location = data_location
        self._channel_weights = channel_weights
        self._input_from_normalized_target = input_from_normalized_target
        if self._channel_weights is None:
            self._channel_weights = [1,1]
        # channel_idx is the key. value is list of full sized frames.
        self._data_dict = load_data(data_type, self._data_location)
        self._frameN = min(len(self._data_dict[0]), len(self._data_dict[1]))
        self._target_channel_idx = target_channel_idx
        self._random_patching = random_patching
        self._uncorrelated_channels = uncorrelated_channels
        self._max_qval = max_qval

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
            normalization_dict = compute_normalization_dict(self._data_dict, self._channel_weights, q_val=self._max_qval, uint8_data=data_type=='cifar10')

        if upper_clip:
            print("Clipping data to {} quantile".format(self._max_qval))
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
        # assert len(self._mean_target) == 2, "mean_target must have length 2"
        # assert len(self._std_target) == 2, "std_target must have length 2"
        self._mean_target = self._mean_target.reshape(-1,1,1)
        self._std_target = self._std_target.reshape(-1,1,1)

        msg = f'[{self.__class__.__name__}] Data: {self._frameN}x{len(self._data_dict.keys())}x{self._data_dict[0][0].shape}'
        msg += f' Patch:{patch_size} Random:{int(random_patching)} Aug:{self._transform is not None} Q:{self._max_qval}'
        if upper_clip is not None:
            msg += f' UpperClip:{int(upper_clip)}'
        msg += f'Uncor:{uncorrelated_channels}'
        if channel_weights is not None:
            msg += f' ChW:{self._channel_weights}'
        
        if self._input_from_normalized_target:
            msg += f' InpFrmNormTar'
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
    
    def patch_location(self, index):
        """
        Returns the frame index along with co-ordinates of the top-left corner of the patch in the frame.
        """
        frame_idx = self.frame_idx(index)
        index = index % self.patch_count_per_frame()
        h,w = self._data_dict[0][frame_idx].shape[-2:]
        h_idx = index // (h//self._patch_size)
        w_idx = index % (w//self._patch_size)
        return frame_idx, h_idx*self._patch_size, w_idx*self._patch_size


    def _get_location(self, index):
        if self._random_patching:
            frame_idx = np.random.randint(0, self._frameN)
            h,w = self._data_dict[0][frame_idx].shape[-2:]
            h_idx = np.random.randint(0, h-self._patch_size) if h > self._patch_size else 0
            w_idx = np.random.randint(0, w-self._patch_size) if w > self._patch_size else 0
        else:
            frame_idx, h_idx, w_idx = self.patch_location(index)
        return frame_idx, h_idx, w_idx
    
    def __getitem__(self, index):

        frame_idx, h_idx, w_idx = self._get_location(index)    
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

        if patch1.ndim == 2:
            patch1 = patch1[None]
            patch2 = patch2[None]

        target = np.concatenate([patch1, patch2], axis=0)
        target = self.normalize_target(target)
        
        if self._input_from_normalized_target:
            inp = self._channel_weights[0]*target[0:1] + self._channel_weights[1]*target[1:2]
        else:
            inp = self._channel_weights[0]*patch1 + self._channel_weights[1]*patch2
            inp = self.normalize_inp(inp)
        

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
    nC = 1 if data_type == 'Hagen' else 3
    uncorrelated_channels = False
    channel_weights = [1,0.3]
    dataset = SplitDataset(data_type, data_location, patch_size, 
                                max_qval=0.98, upper_clip=True,
                             normalization_dict=None, enable_transforms=True,
                             channel_weights=channel_weights,
                             uncorrelated_channels=True, random_patching=True,
                             input_from_normalized_target=True)
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
    _,ax = plt.subplots(figsize=(6,2),ncols=3)
    ax[0].imshow((2+inp.transpose(1,2,0))/4)
    ax[1].imshow((1 +target[:nC].transpose(1,2,0))/2)
    ax[2].imshow((1+target[nC:].transpose(1,2,0))/2)
    # disable axis
    for a in ax:
        a.axis('off')