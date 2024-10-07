import sys; sys.path.append('..')
import numpy as np
from tqdm import tqdm
from data.split_dataset import SplitDataset, compute_normalization_dict, DataLocation

class Normalizer:
    """
    We will use this class to normalize the input images for different values of t
    """
    def __init__(self, data_dict, max_qval, step_size=0.05):
        self.step_size = step_size
        self.alpha_values= np.arange(0.0, 1.0 + step_size, step_size)
        self.normalization_dicts = []
        for alpha in tqdm(self.alpha_values):
            n_dict = compute_normalization_dict(data_dict, [alpha, 1-alpha], q_val=max_qval, uint8_data=False)
            self.normalization_dicts.append(n_dict)

    def get_for_t(self, t):
        s_idx = np.floor(t/self.step_size)
        e_idx = np.ceil(t/self.step_size)
        if s_idx == e_idx:
            return self.normalization_dicts[int(s_idx)]
        else:
            s_dict = self.normalization_dicts[int(s_idx)]
            e_dict = self.normalization_dicts[int(e_idx)]
            w = (t - s_idx*self.step_size)/self.step_size
            # print(s_idx, e_idx,t, w)
            return {k: (1-w)*s_dict[k] + (w)*e_dict[k] for k in s_dict.keys()}
    
    def normalize_input(self, img, t):
        norm_dict = self.get_for_t(t)
        return (img - norm_dict['mean_input'])/norm_dict['std_input']
    
class TimePredictorDataset(SplitDataset):
    def __init__(self, *args, **kwargs):
        super(TimePredictorDataset, self).__init__(*args, **kwargs)
        self.normalization_dicts = []
        self.normalizer = Normalizer(self._data_dict, self._max_qval)

        
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

        t = np.random.rand()
        inp = t*patch1 + (1-t)*patch2
        if inp.ndim == 2:
            inp = inp[None]
        
        inp = self.normalizer.normalize_input(inp,t)
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
    channel_weights = [1,0.3]
    dataset = TimePredictorDataset(data_type, data_location, patch_size, 
                                max_qval=0.98, upper_clip=True,
                             normalization_dict=None, enable_transforms=False,
                             channel_weights=channel_weights,
                             uncorrelated_channels=False, random_patching=False)
    print(len(dataset))
    img, t = dataset[10]
    plt.imshow(img[0])
    print(t)