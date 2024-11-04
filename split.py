import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from data.split_dataset import SplitDataset, DataLocation
from data.rrw_dataset import my_dataset_wTxt as RRWDataset
from data.split_dataset_tiledpred import SplitDatasetTiledPred

from core.psnr import PSNR
from collections import defaultdict
from predtiler.dataset import get_tiling_dataset, get_tile_manager
# from tensorboardX import SummaryWriter
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
    target_channel_idx = opt['datasets'].get('target_channel_idx', None)
    upper_clip = opt['datasets'].get('upper_clip', None)
    max_qval = opt['datasets']['max_qval']
    channel_weights = opt['datasets'].get('channel_weights', None)

    data_type = opt['datasets']['train']['name']  
    uncorrelated_channels = opt['datasets']['train']['uncorrelated_channels']
    assert data_type in ['cifar10', 'Hagen', "RRW"], "Only cifar10, Hagen and RRW datasets are supported"
    if data_type == 'RRW':
        rootdir = opt['datasets']['datapath']
        train_fpath = os.path.join(rootdir, 'train.txt')
        val_fpath = os.path.join(rootdir, 'val.txt')
        datapath = os.path.join(rootdir, 'RRWDatasets')
        train_set = RRWDataset(datapath, train_fpath, crop_size=patch_size, fix_sample_A=1e10, regular_aug=True)
        val_set = RRWDataset(datapath, val_fpath, crop_size=patch_size, fix_sample_A=1e10, regular_aug=False)
        return train_set, val_set
    else:
        if data_type == 'Hagen':
            train_data_location = DataLocation(channelwise_fpath=(opt['datasets']['train']['datapath']['ch0'],
                                                            opt['datasets']['train']['datapath']['ch1']))
            val_data_location = DataLocation(channelwise_fpath=(opt['datasets']['val']['datapath']['ch0'],
                                                            opt['datasets']['val']['datapath']['ch1']))
        elif data_type == 'cifar10':
            train_data_location = DataLocation(directory=(opt['datasets']['train']['datapath']))
            val_data_location = DataLocation(directory=(opt['datasets']['val']['datapath']))
        
        input_from_normalized_target = opt['model']['which_model_G'] == 'joint_indi'
        train_set = SplitDataset(data_type, train_data_location, patch_size, 
                                target_channel_idx=target_channel_idx, 
                                    max_qval=max_qval, upper_clip=upper_clip,
                                    uncorrelated_channels=uncorrelated_channels,
                                    channel_weights=channel_weights,
                                normalization_dict=None, enable_transforms=True,random_patching=True, input_from_normalized_target=input_from_normalized_target)

        if not tiled_pred:
            class_obj = SplitDataset 
        else:
            data_shape = (10, 2048, 2048)
            tile_manager = get_tile_manager(data_shape, (1, patch_size//2, patch_size//2), (1, patch_size, patch_size))
            class_obj = get_tiling_dataset(SplitDataset, tile_manager)

        val_set = class_obj(data_type, val_data_location, patch_size, target_channel_idx=target_channel_idx,
                            normalization_dict=train_set.get_normalization_dict(),
                            max_qval=max_qval,
                                upper_clip=upper_clip,
                                channel_weights=channel_weights,
                            enable_transforms=False,
                                                        random_patching=False, input_from_normalized_target=input_from_normalized_target)
        return train_set, val_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-rootdir', type=str, default='/group/jug/ashesh/training/diffsplit')
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    #sanity checks
    model_conf = opt['model'] 
    assert model_conf['unet']['out_channel'] == model_conf['diffusion']['channels']
    # assert model_conf['unet']['in_channel'] == 1 + model_conf['unet']['out_channel'], "Input channel= concat([noise, input]) and noise has same shape as target"

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

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
    train_loader = Data.create_dataloader(train_set, opt['datasets']['train'], 'train')
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')

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

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
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
                        # tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                psnr_values= defaultdict(list)
                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        if idx == 20:
                            break
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        # input, target, prediction = unnormalize_data(visuals,train_set.get_normalization_dict())
                        input = visuals['input'].cpu().numpy()
                        target = visuals['target'].cpu().numpy()
                        prediction = visuals['prediction'].cpu().numpy()
                        # input_img = Metrics.tensor2img(input, min_max=[input.min(), input.max()])  # uint8
                        target_arr = []
                        pred_arr = []
                        mean_target = val_set.get_normalization_dict()['mean_target']
                        std_target = val_set.get_normalization_dict()['std_target']
                        mean_input = val_set.get_normalization_dict()['mean_input']
                        std_input = val_set.get_normalization_dict()['std_input']
                        assert input.shape[0] == 1
                        assert target.shape[0] == 1
                        assert prediction.shape[0] == 1
                        input = input[0]
                        target = target[0]
                        prediction = prediction[0]
                        input_img = ((input * std_input + mean_input)/2).astype(np.uint16)
                        target_img = (target * std_target + mean_target).astype(np.uint16)
                        pred_img = (prediction * std_target + mean_target)
                        pred_img[pred_img < 0] = 0
                        pred_img[pred_img > 65535] = 65535
                        pred_img=pred_img.astype(np.uint16)
                        mode = 'RGB' if input.shape[0] == 3 else 'L'
                        
                        ncols = 3 if mode == 'RGB' else 1
                        for ch_idx in range(0,target.shape[0],ncols):
                            psnr_val = PSNR(target_img[ch_idx:ch_idx+ncols]*1.0, pred_img[ch_idx:ch_idx+ncols]*1.0).mean().item()
                            psnr_values[ch_idx].append(psnr_val)
                        # if wandb_logger:
                        #     wandb_logger.log_image(
                        #         f'validation_{idx}', 
                        #         np.concatenate((pred_img, target_img), axis=1)
                        #     )
                        if mode != 'RGB':
                            # it is uint16. it is better to normalize it to 0-1
                            minv = target_img.reshape(target_img.shape[0],-1).min(axis=1).reshape(-1,1,1)
                            target_img = target_img - minv
                            maxv = target_img.reshape(target_img.shape[0],-1).max(axis=1).reshape(-1,1,1)
                            target_img = target_img / maxv

                            input_img = input_img - input_img.min()
                            max_val_input = input_img.reshape(input_img.shape[0],-1).max(axis=1).reshape(-1,1,1)
                            input_img = input_img / max_val_input

                            pred_img = pred_img - minv
                            pred_img = pred_img / maxv
                            pred_img[pred_img < 0] = 0
                            pred_img[pred_img > 1] = 1


                        # print(target_img.max(), target_img.min(), input_img.max(), input_img.min(), pred_img.max(), pred_img.min())
                        # generation
                        Metrics.save_img(
                            target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx), mode=mode)
                        Metrics.save_img(
                            input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx), mode=mode)
                        Metrics.save_img(pred_img, '{}/{}_{}_pred.png'.format(result_path, current_step, idx), mode=mode)


                    avg_psnr = np.mean([np.mean(psnr_values[ch_idx]) for ch_idx in psnr_values.keys()])
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    # tb_logger.add_scalar('psnr', avg_psnr, current_step)

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
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
