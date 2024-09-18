from unittest.mock import Mock

import numpy as np
# import sys; sys.path.append('../')
from data.tile_stitcher import stitch_predictions
from data.tiling_manager import TilingMode
from data.split_dataset_tiledpred import SplitDatasetTiledPred


def get_data(*args,**kwargs):
    n = 5
    H = 512
    W = 512
    C = 2
    data = np.arange(n*H*W*C).reshape(n,H,W,C)
    return {i:data[...,i] for i in range(C)}


def identity_normalization_dict():
        # self._mean_inp = normalization_dict['mean_input']
        # self._std_inp = normalization_dict['std_input']
        # self._mean_target = normalization_dict['mean_target']
        # self._std_target = normalization_dict['std_target']
        # self._target0_max = normalization_dict['target0_max']
        # self._target1_max = normalization_dict['target1_max']
        # self._input_max = normalization_dict['input_max']
        return {'mean_input':0,
                'mean_target': np.array([0,0]),
                'std_input':1,
                'std_target': np.array([1,1]),
                'target0_max': 1,
                'target1_max': 1,
                'input_max': 1}

def test_stich_prediction(monkeypatch):
    monkeypatch.setattr('data.split_dataset.load_data', get_data)
    data_location = None
    patch_size = 256
    grid_size = 128
    data_type = 'Hagen'
    dset = SplitDatasetTiledPred(data_type, data_location, patch_size, 
                                    grid_size=grid_size,
                                    upper_clip=False,
                             normalization_dict=identity_normalization_dict(), enable_transforms=False,
                             uncorrelated_channels=False, random_patching=False)
    
    predictions = []
    for i in range(len(dset)):
        predictions.append(dset[i]['target'])
    
    predictions = np.stack(predictions)
    stitched_pred = stitch_predictions(predictions, dset.tile_manager)
    actual_data = get_data()
    actual_data = np.concatenate([actual_data[i][...,None] for i in range(len(actual_data.keys()))], axis=-1)
    assert (stitched_pred== actual_data).all()
