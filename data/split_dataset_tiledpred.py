"""
This dataset class can be used for tiled prediction.
"""
import sys; sys.path.append('..')

from data.split_dataset import SplitDataset, DataLocation
from data.tiling_manager import TileIndexManager, TilingMode

class SplitDatasetTiledPred(SplitDataset):
    def __init__(self, *args, **kwargs):
        grid_size = kwargs.pop('grid_size', None)
        super().__init__(*args, **kwargs)
        if grid_size is None:
            grid_size = self._patch_size//2
        
        nC = len(self._data_dict.keys())
        H, W  = self._data_dict[0][0].shape
        # breakpoint()
        patch_shape = (1,self._patch_size, self._patch_size)
        grid_shape = (1, grid_size, grid_size)
        data_shape = (self._frameN, H, W)
        tiling_mode = TilingMode.ShiftBoundary
        self.tile_manager = TileIndexManager(data_shape, grid_shape, patch_shape, tiling_mode)


    def __len__(self):
        return self.tile_manager.total_grid_count()

    def patch_loc(self, index):
        patch_loc_list = self.tile_manager.get_patch_location_from_dataset_idx(index)
        print(patch_loc_list)
        return patch_loc_list
    


if __name__ == "__main__":
    import sys
    data_location = DataLocation(channelwise_fpath=('/group/jug/ashesh/data/diffsplit_hagen/val/val_actin-60x-noise2-highsnr.tif',
                                                    '/group/jug/ashesh/data/diffsplit_hagen/val/val_mito-60x-noise2-highsnr.tif'))
    # patch_size = 512
    # data_type = 'hagen'
    # data_location = DataLocation(directory='/group/jug/ashesh/data/cifar-10-python/train')
    patch_size = 256
    grid_size = 128
    data_type = 'Hagen'
    nC = 1 if data_type == 'Hagen' else 3
    uncorrelated_channels = False
    dataset = SplitDatasetTiledPred(data_type, data_location, patch_size, 
                                    grid_size=grid_size,
                                max_qval=0.98, upper_clip=True,
                             normalization_dict=None, enable_transforms=False,
                             uncorrelated_channels=False, random_patching=False)
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        inp = data['input']
        target = data['target']
        # print(inp.min(), inp.max(),end='\t')
        # print(target[0].min(), target[0].max(), end='\t')
        # print(target[1].min(), target[1].max())
        # break   


    import matplotlib.pyplot as plt
    data= dataset[0]
    inp = data['input']
    target = data['target']
    _,ax = plt.subplots(figsize=(3,1),ncols=3)
    ax[0].imshow((2+inp.transpose(1,2,0))/4)
    ax[1].imshow((1 +target[:nC].transpose(1,2,0))/2)
    ax[2].imshow((1+target[nC:].transpose(1,2,0))/2)
    # disable axis
    for a in ax:
        a.axis('off')