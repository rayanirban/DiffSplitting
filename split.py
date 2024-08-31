import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from data.split_dataset import SplitDataset, DataLocation
from core.psnr import PSNR
from collections import defaultdict
# from tensorboardX import SummaryWriter
import os
import numpy as np

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
    model_conf['unet']['out_channel'] == model_conf['diffusion']['channels']
    model_conf['unet']['in_channel'] == 1 + model_conf['unet']['out_channel'], "Input channel= concat([noise, input]) and noise has same shape as target"
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt, opt['path']['experiment_root'], opt['experiment_name'])
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    patch_size = opt['datasets']['patch_size']
    target_channel_idx = opt['datasets']['target_channel_idx']
    upper_clip = opt['datasets']['upper_clip']
    max_qval = opt['datasets']['max_qval']
    
    data_type = opt['datasets']['train']['name']    
    assert data_type in ['cifar10', 'Hagen']
    if data_type == 'Hagen':
        train_data_location = DataLocation(channelwise_fpath=(opt['datasets']['train']['datapath']['ch0'],
                                                        opt['datasets']['train']['datapath']['ch1']))
        val_data_location = DataLocation(channelwise_fpath=(opt['datasets']['val']['datapath']['ch0'],
                                                        opt['datasets']['val']['datapath']['ch1']))
    elif data_type == 'cifar10':
        train_data_location = DataLocation(directory=(opt['datasets']['train']['datapath']))
        val_data_location = DataLocation(directory=(opt['datasets']['val']['datapath']))
    
    train_set = SplitDataset(data_type, train_data_location, patch_size, 
                             target_channel_idx=target_channel_idx, 
                                max_qval=max_qval, upper_clip=upper_clip,
                             normalization_dict=None, enable_transforms=True,random_patching=True)
    train_loader = Data.create_dataloader(train_set, opt['datasets']['train'], 'train')

    val_set = SplitDataset(data_type, val_data_location, patch_size, target_channel_idx=target_channel_idx,
                           normalization_dict=train_set.get_normalization_dict(),
                           max_qval=max_qval,
                            upper_clip=upper_clip,
                           enable_transforms=False,
                                                     random_patching=False)
    val_loader = Data.create_dataloader(val_set, opt['datasets']['val'], 'val')
    # dataset
    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase == 'train' and args.phase != 'val':
    #         train_set = Data.create_dataset(dataset_opt, phase)
    #         train_loader = Data.create_dataloader(
    #             train_set, dataset_opt, phase)
    #     elif phase == 'val':
    #         val_set = Data.create_dataset(dataset_opt, phase)
    #         val_loader = Data.create_dataloader(
    #             val_set, dataset_opt, phase)
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
                        message += '{:s}: {:.4e} '.format(k, v)
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
                        input = input[0]
                        target = target[0]
                        input_img = (input * std_input + mean_input).astype(np.uint16)
                        target_img = (target * std_target + mean_target).astype(np.uint16)
                        pred_img = (prediction * std_target + mean_target).astype(np.uint16)
                        
                        mode = 'RGB' if input.shape[0] == 3 else 'L'
                        # generation
                        Metrics.save_img(
                            target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
                        Metrics.save_img(pred_img, '{}/{}_{}_pred.png'.format(result_path, current_step, idx))
                        
                        for ch_idx in range(target.shape[0]):
                            psnr_values[ch_idx].append(PSNR(target_img[ch_idx][None]*1.0, pred_img[ch_idx][None]*1.0))

                        # if wandb_logger:
                        #     wandb_logger.log_image(
                        #         f'validation_{idx}', 
                        #         np.concatenate((pred_img, target_img), axis=1)
                        #     )

                    avg_psnr = avg_psnr / idx
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
