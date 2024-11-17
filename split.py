import torch
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from data.rrw_dataset import my_dataset_wTxt as RRWDataset
from data.data_dehaze import Dataset
from pathlib import Path
from core.psnr import PSNR, RangeInvariantPsnr
from collections import defaultdict
import os
import numpy as np
import git

def add_git_info(opt):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(dir_path, search_parent_directories=True)
    opt['git'] = {}
    opt['git']['changedFiles'] = [item.a_path for item in repo.index.diff(None)]
    opt['git']['branch'] = repo.active_branch.name
    opt['git']['untracked_files'] = repo.untracked_files
    opt['git']['latest_commit'] = repo.head.object.hexsha


def get_datasets(opt, tiled_pred=False):
    patch_size = opt['datasets']['patch_size']
    data_type = opt['datasets']['train']['name']  
    assert data_type in ['cifar10', 'Hagen', "RRW", "Neuron"], "Only cifar10, Hagen and RRW datasets are supported"
    if data_type == 'RRW':
        rootdir = opt['datasets']['datapath']
        train_fpath = os.path.join(rootdir, 'train.txt')
        val_fpath = os.path.join(rootdir, 'val.txt')
        datapath = os.path.join(rootdir, 'RRWDatasets')
        nimgs = 2000
        train_set = RRWDataset(datapath, train_fpath, crop_size=patch_size, fix_sample_A=nimgs, regular_aug=True)
        val_set = RRWDataset(datapath, val_fpath, crop_size=patch_size, fix_sample_A=nimgs, regular_aug=False)
        return train_set, val_set
    elif data_type == 'Neuron':
        dir_img_noisy_train = '/group/jug/Anirban/Datasets/AllNeuron_Combined_55_mid/train_noisy/'
        dir_img_clean_train = '/group/jug/Anirban/Datasets/AllNeuron_Combined_55_mid/train_clean/'
        dir_img_noisy_val = '/group/jug/Anirban/Datasets/AllNeuron_Combined_55_mid/val_noisy/'
        dir_img_clean_val = '/group/jug/Anirban/Datasets/AllNeuron_Combined_55_mid/val_clean/'
        train_set = Dataset(folder_noisy = Path(dir_img_noisy_train), folder_clean = Path(dir_img_clean_train), returns=[0, 3], returns_type=['c','n'], mode='train')
        val_set = Dataset(folder_noisy= Path(dir_img_noisy_val), folder_clean = Path(dir_img_clean_val), returns=[0, 3], returns_type=['c','n'], mode='val')
        return train_set, val_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-rootdir', type=str, default='/group/jug/Anirban/training/dehaze/')
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    #sanity checks
    model_conf = opt['model'] 
    assert model_conf['unet']['out_channel'] == model_conf['diffusion']['channels']

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        add_git_info(opt)
        wandb_logger = WandbLogger(opt, opt['path']['experiment_root'], opt['experiment_name'])
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None


    train_set, val_set = get_datasets(opt)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.2e} '.format(k, v)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                psnr_values= defaultdict(list)
                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        if idx == 20:
                            break
                        diffusion.feed_data(val_data)
                        diffusion.test(continuous=False)
                        visuals = diffusion.get_current_visuals()

                        input = visuals['input'].cpu().numpy()
                        target = visuals['target'].cpu().numpy()
                        prediction = visuals['prediction'].cpu().numpy()

                        assert input.shape[0] == 1
                        assert target.shape[0] == 1
                        assert prediction.shape[0] == 1

                        input = input[0]
                        target = target[0]
                        prediction = prediction[0]

                        input_img = input.astype(np.float32)
                        target_img = target.astype(np.float32)
                        pred_img = prediction.astype(np.float32)

                        mode = 'TIFF'
                        
                        print('input shape:', input_img.shape)
                        print('target shape:', target_img.shape)
                        print('pred shape:', pred_img.shape)

                        psnr_val = RangeInvariantPsnr(target_img, pred_img).mean().item()
                        psnr_values[idx].append(psnr_val)

                        Metrics.save_tif(target_img, '{}/{}_{}_target.tif'.format(result_path, current_step, idx))
                        Metrics.save_tif(input_img, '{}/{}_{}_input.tif'.format(result_path, current_step, idx))
                        Metrics.save_tif(pred_img, '{}/{}_{}_pred.tif'.format(result_path, current_step, idx))

                    avg_psnr = np.mean([np.mean(psnr_values[idx]) for idx in psnr_values.keys()])
                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR']).astype(np.float32)
            lr_img = Metrics.tensor2img(visuals['LR']).astype(np.float32)
            fake_img = Metrics.tensor2img(visuals['INF']).astype(np.float32)

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                sr_img = visuals['SR']
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(Metrics.tensor2img(sr_img[iter]).astype(np.float32), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                sr_img = Metrics.tensor2img(visuals['SR']).astype(np.float32)
                Metrics.save_img(sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(Metrics.tensor2img(visuals['SR'][-1]).astype(np.float32), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]).astype(np.float32), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]).astype(np.float32), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]).astype(np.float32), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })