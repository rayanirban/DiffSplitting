from collections import defaultdict
import os
import numpy as np

def get_raw_textfpaths(rootdir):
    fpaths = [
    os.path.join(rootdir,'DeRef_HZ1-S1_4.txt'),
    os.path.join(rootdir,'DeRef_HZ5.txt'),
    os.path.join(rootdir,'DeRef_HZ1.txt'),
    os.path.join(rootdir,'DeRef_HZ6-S1_4.txt'),
    os.path.join(rootdir,'DeRef_HZ2-S1_4.txt'),
    os.path.join(rootdir,'DeRef_HZ6.txt'),
    os.path.join(rootdir,'DeRef_HZ2.txt'),
    os.path.join(rootdir,'DeRef_camera1-S1_4.txt'),
    os.path.join(rootdir,'DeRef_HZ3-S1_4.txt'),
    os.path.join(rootdir,'DeRef_camera1.txt'),
    os.path.join(rootdir,'DeRef_HZ3.txt'),
    os.path.join(rootdir,'DeRef_camera2.txt'),
    os.path.join(rootdir,'DeRef_HZ4-S1_4.txt'),
    os.path.join(rootdir,'DeRef_hf0.txt'),
    os.path.join(rootdir,'DeRef_HZ4.txt'),
    os.path.join(rootdir,'DeRef_hf1.txt'),
    os.path.join(rootdir,'DeRef_HZ5-S1_4.txt'),
    os.path.join(rootdir,'DeRef_hf2.txt'),
    ]
    return fpaths


def train_val_test_split(rootdir, val_fraction=0.1, test_fraction=0.1):
    textfpaths = get_raw_textfpaths(rootdir)
    train_data = {}
    val_data = {}
    test_data = {}
    for textfpath in textfpaths:
        data_dict = load_paired_fnames(textfpath)
        gt_keys = np.random.RandomState(seed=955).permutation(sorted(list(data_dict.keys())))
        num_val = int(len(gt_keys) * val_fraction)
        num_test = int(len(gt_keys) * test_fraction)
        num_train = len(gt_keys) - num_val - num_test
        train_dict = {k: data_dict[k] for k in gt_keys[:num_train]}
        val_dict = {k: data_dict[k] for k in gt_keys[num_train:num_train+num_val]}
        test_dict = {k: data_dict[k] for k in gt_keys[num_train+num_val:]}

        train_data.update(train_dict)
        val_data.update(val_dict)
        test_data.update(test_dict)

    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    return train_data, val_data, test_data


def load_paired_fnames(fpath):
    """
    Returns a dicttionary of the following format:
        GT_fname: [input_fname1, input_fname2, ...]
    """
    with open(fpath) as file:
        lines = [line.rstrip().replace("//","/") for line in file]
        data_dict = defaultdict(list)
        for line in lines:
            inpfname, targetfname = line.split(' ')
            data_dict[targetfname].append(inpfname)
    return data_dict    

def save_train_val_test_split(outputdir, textfilesdir):
    """
    Creates three text files: train.txt, val.txt, test.txt
    They contain the following format:
    input_fname1 GT_fname1
    """
    train_data, val_data, test_data = train_val_test_split(textfilesdir)
    train_fpath = os.path.join(outputdir, 'train.txt')
    trainCounter = 0
    with open(train_fpath, 'w') as file:
        for k, v in train_data.items():
            for inp in v:
                file.write(f"{inp} {k}\n")
                trainCounter += 1
    print(f"Saved train data N:{trainCounter} to {train_fpath}")

    val_fpath = os.path.join(outputdir, 'val.txt')
    valCounter = 0
    with open(val_fpath, 'w') as file:
        for k, v in val_data.items():
            for inp in v:
                file.write(f"{inp} {k}\n")
                valCounter += 1
    print(f"Saved val data N:{valCounter} to {val_fpath}")

    
    test_fpath = os.path.join(outputdir, 'test.txt')    
    testCounter = 0
    with open(test_fpath, 'w') as file:
        for k, v in test_data.items():
            for inp in v:
                file.write(f"{inp} {k}\n")
                testCounter += 1
    print(f"Saved test data N:{testCounter} to {test_fpath}")

if __name__ == '__main__':
    outputdir = '/group/jug/ashesh/data/RRW/combined/'
    textfilesdir = '/group/jug/ashesh/data/RRW/combined/RRWDatasets/'
    save_train_val_test_split(outputdir, textfilesdir)
