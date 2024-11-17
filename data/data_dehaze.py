import torch
import torch.nn.functional as F
from torch.utils import data
from pathlib import Path
import numpy as np
from tqdm import tqdm
from tifffile import imread
from .util_dehaze import normalize_image_clean, normalize_image_noisy, normalize_haze_clean, normalize_haze_noisy

class Dataset(data.Dataset):
    def __init__(self, folder_noisy = Path('/group/jug/Anirban/Datasets/AllNeuron_Combined_55_mid/train_noisy/'), \
                 folder_clean = Path('/group/jug/Anirban/Datasets/AllNeuron_Combined_55_mid/train_clean/'), \
                    start = 0, step = 6, image_size=64, \
                 returns = [0,1,2,3], returns_type = ['c', 'c', 'c', 'n'], mode = 'train', exts = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp']):
        super().__init__()
        self.folder_noisy = folder_noisy
        self.folder_clean = folder_clean
        self.image_size = image_size
        self.paths_noisy = [p for ext in exts for p in Path(f'{folder_noisy}').glob(f'**/*.{ext}')]
        self.paths_clean = [p for ext in exts for p in Path(f'{folder_clean}').glob(f'**/*.{ext}')]
        self.start = start
        self.step = step
        self.mode = mode
        self.returns = returns
        self.returns_type = returns_type

    def __len__(self):
        return len(self.paths_noisy) # returns the number of images in the dataset

    def __getitem__(self, index):

        paths_clean = self.paths_clean[index]
        paths_noisy = self.paths_noisy[index]
        
        img_clean = imread(paths_clean)
        img_clean = img_clean[self.start::self.step][0:1]
        img_clean = img_clean.astype(np.float32)
        img_clean = torch.from_numpy(img_clean.copy()) 

        img_noisy = imread(paths_noisy)
        img_noisy = img_noisy[self.start::self.step][3:4]
        img_noisy = img_noisy.astype(np.float32)
        img_noisy = torch.from_numpy(img_noisy.copy())
        
        image_clean = normalize_image_clean(img_clean, number=0)
        image_noisy = normalize_image_noisy(img_noisy, number=3)

        data = {'target': image_clean, 'input': image_noisy}        

        return data