import os
import pickle
import numpy as np
from collections import defaultdict

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def training_files(datadir):
    # return all files
    return os.listdir(datadir)

def testing_files():
    return ['test_batch']

def load_cifar10_data(fpath):
    print(fpath)
    data = unpickle(fpath)
    imgs = data[b'data'].reshape(-1, 3, 32, 32)
    labels = data[b'labels']
    return imgs, labels

def load_train_val_data(datadir,label_idx_list):
    fnames = training_files(datadir)
    fpaths = [os.path.join(datadir, f) for f in fnames]
    data =defaultdict(list)

    for fpath in fpaths:
        imgs, labels = load_cifar10_data(fpath)
        labels = np.array(labels)
        for i in range(len(label_idx_list)):
            idx = np.where(labels == label_idx_list[i])[0]
            data[i].append(imgs[idx])
    
    for i in range(len(label_idx_list)):
        data[i] = np.concatenate(data[i], axis=0)
    
    return data
    




if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    datadir = '/group/jug/ashesh/data/cifar-10-python/train'
    fpath = os.path.join(datadir, 'data_batch_1')
    imgs, labels = load_cifar10_data(fpath)
    labels = np.array(labels)
    _,ax = plt.subplots(figsize=(15,1.5),ncols=10)

    for i in range(10):
        idx_list = np.where(labels == i)[0]
        idx = np.random.choice(idx_list)
        ax[i].imshow(imgs[idx].transpose(1,2,0))
        ax[i].axis('off')