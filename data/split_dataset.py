import numpy as np
import albumentations as A
import os
from skimage.io import imread
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DataLocation:
    fpath: str = ''
    channelwise_fpath: Tuple[str]= ()
    directory: str = ''

    def __post_init__(self):
        assert self.fpath or len(self.channelwise_fpath) or self.directory, "At least one of the following must be provided: fpath, channelwise_fpath, directory"
        assert (self.fpath and not self.channelwise_fpath and not self.directory) or (not self.fpath and self.channelwise_fpath and not self.directory) or (not self.fpath and not self.channelwise_fpath and self.directory), "Only one of the following must be provided: fpath, channelwise_fpath, directory"

def load_data(dataloc:DataLocation):
    if dataloc.fpath:
        return _load_data_fpath(dataloc.fpath)
    elif len(dataloc.channelwise_fpath) > 0:
        return _load_data_channelwise_fpath(dataloc.channelwise_fpath)

def compute_normalization_dict(data_dict):
    tar1_mean = np.mean([np.mean(x) for x in data_dict[0]])
    tar2_mean = np.mean([np.mean(x) for x in data_dict[1]])
    tar1_std = np.mean([np.std(x) for x in data_dict[0]])
    tar2_std = np.mean([np.std(x) for x in data_dict[1]])
    inp_mean = (tar1_mean + tar2_mean)
    inp_std = (tar1_std**2 + tar2_std**2)**0.5
    return {
        'mean_input': inp_mean,
        'std_input': inp_std,
        'mean_target': np.array([tar1_mean, tar2_mean]),
        'std_target': np.array([tar1_std, tar2_std])
    }

def _load_data_channelwise_fpath(fpaths:Tuple[str]):
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
    def __init__(self, data_location:DataLocation, patch_size, target_channel_idx = None,random_patching=False, 
                 enable_transforms=False,
                 normalization_dict=None):

        self._patch_size = patch_size
        self._data_location = data_location

        # channel_idx is the key. value is list of full sized frames.
        self._data_dict = load_data(self._data_location)
        self._frameN = len(self._data_dict[0])
        self._target_channel_idx = target_channel_idx
        self._random_patching = random_patching

        self._transform = None
        if enable_transforms:
            self._transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)],
                additional_targets={'image2': 'image'})

        if normalization_dict is None:
            print("Computing mean and std for normalization")
            normalization_dict = compute_normalization_dict(self._data_dict)

        assert 'mean_input' in normalization_dict, "mean_input must be provided"
        assert 'std_input' in normalization_dict, "std_input must be provided"
        assert 'mean_target' in normalization_dict, "mean_target must be provided"
        assert 'std_target' in normalization_dict, "std_target must be provided"

        self._mean_inp = normalization_dict['mean_input']
        self._std_inp = normalization_dict['std_input']
        self._mean_target = normalization_dict['mean_target']
        self._std_target = normalization_dict['std_target']

        assert isinstance(self._mean_target, np.ndarray), "mean_target must be a numpy array"
        assert isinstance(self._std_target, np.ndarray), "std_target must be a numpy array"
        assert len(self._mean_target) == 2, "mean_target must have length 2"
        assert len(self._std_target) == 2, "std_target must have length 2"
        self._mean_target = self._mean_target.reshape(2,1,1)
        self._std_target = self._std_target.reshape(2,1,1)
        print(f'[{self.__class__.__name__}] Data: {self._frameN}x{len(self._data_dict.keys())}x{self._data_dict[0][0].shape} \
              Patch:{patch_size} Random:{int(random_patching)} Aug:{self._transform is not None}')

    def get_normalization_dict(self):
        assert self._mean_inp is not None, "Mean and std have not been computed"
        
        return {
            'mean_input': self._mean_inp,
            'std_input': self._std_inp,
            'mean_target': self._mean_target,
            'std_target': self._std_target
        }
    def normalize_inp(self, inp):
        norm_inp = (inp - self._mean_inp)/self._std_inp
        return norm_inp.astype(np.float32)
    
    def normalize_target(self, target):
        norm_tar = (target - self._mean_target)/self._std_target
        return norm_tar.astype(np.float32)
    
    def patch_count_per_frame(self):
        h,w = self._data_dict[0][0].shape
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
        h,w = self._data_dict[0][frame_idx].shape
        h_idx = index // (h//self._patch_size)
        w_idx = index % (w//self._patch_size)
        return frame_idx, h_idx*self._patch_size, w_idx*self._patch_size


    def __getitem__(self, index):
        if self._random_patching:
            frame_idx = np.random.randint(0, self._frameN)
            h,w = self._data_dict[0][frame_idx].shape
            h_idx = np.random.randint(0, h-self._patch_size)
            w_idx = np.random.randint(0, w-self._patch_size)
        else:
            frame_idx, h_idx, w_idx = self.patch_loc(index)
        
        img1 = self._data_dict[0][frame_idx]
        img2 = self._data_dict[1][frame_idx]
        assert img1.shape == img2.shape, "Images must have the same shape"
        # random h,w location
        patch1 = img1[h_idx:h_idx+self._patch_size, w_idx:w_idx+self._patch_size]
        patch2 = img2[h_idx:h_idx+self._patch_size, w_idx:w_idx+self._patch_size]
        if self._transform:
            transformed = self._transform(image=patch1, image2=patch2)
            patch1 = transformed['image']
            patch2 = transformed['image2']

        inp = patch1 + patch2
        inp = inp[None]
        target = np.stack([patch1, patch2], axis=0)
        
        inp = self.normalize_inp(inp)
        target = self.normalize_target(target)
        if self._target_channel_idx is None:
            return {'input':inp, 'target':target}
        
        return {'input':inp, 'target':target[self._target_channel_idx: self._target_channel_idx+1]}
    

if __name__ == "__main__":
    data_location = DataLocation(channelwise_fpath=('/group/jug/ashesh/data/ventura_gigascience_small/actin-60x-noise2-highsnr.tif',
                                                    '/group/jug/ashesh/data/ventura_gigascience_small/mito-60x-noise2-highsnr.tif'))
    patch_size = 512
    dataset = SplitDataset(data_location, patch_size, normalization_dict=None)
    print(len(dataset))
    data = dataset[0]
    inp = data['input']
    target = data['target']
    print(inp.shape, target.shape)
    print(inp.mean(), inp.std())
    print(target.mean(), target.std())