import sys; sys.path.append('..')
import numpy as np
from tqdm import tqdm
from data.split_dataset import SplitDataset, compute_normalization_dict, DataLocation

def compute_input_normalization_dict(data_dict, n_timesteps, mean_target, std_target):
    mean_ch0, mean_ch1 = mean_target.squeeze()
    std_ch0, std_ch1 = std_target.squeeze()
    ch0 = [(x - mean_ch0)/std_ch0 for x in data_dict[0]]
    ch1 = [(x - mean_ch1)/std_ch1 for x in data_dict[1]]
    output = {}
    for t_int in tqdm(np.arange(0,n_timesteps+1)):
        t = t_int/n_timesteps
        ch_min = 1e10
        ch_max = -1e10
        for idx in range(len(ch0)):
            ch = t*ch0[idx] + (1-t)*ch1[idx]
            ch_min = min(ch_min, ch.min())
            ch_max = max(ch_max, ch.max())
        output[t_int] = [ch_min, ch_max]
    return output
    

class TimePredictorDataset(SplitDataset):
    def __init__(self, *args, **kwargs):
        if 'step_size' in kwargs:
            step_size = kwargs.pop('step_size')
        else:
            step_size = 0.05
        
        if 'gaussian_noise_std_factor' in kwargs:
            self._gaussian_noise_std_factor = kwargs.pop('gaussian_noise_std_factor')
        else:
            self._gaussian_noise_std_factor = None
        super(TimePredictorDataset, self).__init__(*args, **kwargs)
        self._num_timesteps = 100
        self.input_normalization_dict = compute_input_normalization_dict(self._data_dict, self._num_timesteps, self._mean_target, self._std_target)
        # self.normalizer = Normalizer(self._data_dict, self._max_qval, step_size= step_size)
        if self._gaussian_noise_std_factor is not None:
            print("Adding Gaussian noise with std factor: ", self._gaussian_noise_std_factor)
        
    def sample_t(self):
        t_int = np.random.randint(0, self._num_timesteps)
        return t_int/self._num_timesteps, t_int

    def min_max_normalize(self, img, t_int):
        t_min, t_max = self.input_normalization_dict[t_int]
        return 2*(img - t_min)/(t_max - t_min) -1
    
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

        # normalize the target 
        target = np.stack([patch1, patch2], axis=0)
        target = self.normalize_target(target)
        patch1, patch2 = target[0], target[1]
        
        t, t_int = self.sample_t()
        inp = t*patch1 + (1-t)*patch2
        inp = self.min_max_normalize(inp, t_int)

        if inp.ndim == 2:
            inp = inp[None]
        
        if self._gaussian_noise_std_factor is not None:
            inp += np.random.normal(0, self._gaussian_noise_std_factor*inp.std(), inp.shape)
        
        return inp, t

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt 
    data_location = DataLocation(channelwise_fpath=('/group/jug/ashesh/data/diffsplit_hagen/val/val_actin-60x-noise2-highsnr.tif',
                                                    '/group/jug/ashesh/data/diffsplit_hagen/val/val_mito-60x-noise2-highsnr.tif'))
    # patch_size = 512
    # data_type = 'hagen'
    # data_location = DataLocation(directory='/group/jug/ashesh/data/cifar-10-python/train')
    patch_size = 512
    data_type = 'Hagen'
    nC = 1 if data_type == 'Hagen' else 3
    channel_weights = [1,1.0]
    dataset = TimePredictorDataset(data_type, data_location, patch_size, 
                                max_qval=0.98, upper_clip=True,
                             normalization_dict=None, enable_transforms=False,
                             channel_weights=channel_weights,
                             uncorrelated_channels=False, random_patching=False,
                             step_size=0.25)
    for i in range(len(dataset)):
        img, t = dataset[i]
        if img.max() > 0:
            print(img.min(),img.max(), t)
    
    print(len(dataset))
    img, t = dataset[10]
    plt.imshow(img[0])
    print(t)