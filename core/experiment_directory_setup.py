"""
This file contains the setup which creates the root directory for the experiment.
"""
from datetime import datetime
import os
from pathlib import Path
import time

def get_new_model_version(model_dir: str) -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(f'Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed')
            exit()
    if len(versions) == 0:
        return '0'
    return f'{max(versions) + 1}'


def get_model_name(config):
    mtype = config["model"]["which_model_G"]
    dtype = config["datasets"]["train"]["name"]
    ltype = config["model"]["loss_type"]
    return f'{dtype}-{mtype}-{ltype}'


def get_month():
    return datetime.now().strftime("%y%m")


def get_workdir(config, root_dir, use_max_version, nested_call=0):
    """
    2408/sr3-hagen-vanilla/5/
    """
    rel_path = get_month()
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_model_name(config))
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    if use_max_version:
        # Used for debugging.
        version = int(get_new_model_version(cur_workdir))
        if version > 0:
            version = f'{version - 1}'

        rel_path = os.path.join(rel_path, str(version))
    else:
        rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))

    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(
            f'Workdir {cur_workdir} already exists. Probably because someother program also created the exact same directory. Trying to get a new version.'
        )
        time.sleep(2.5)
        if nested_call > 10:
            raise ValueError(f'Cannot create a new directory. {cur_workdir} already exists.')

        return get_workdir(config, root_dir, use_max_version, nested_call + 1)

    return cur_workdir, rel_path
