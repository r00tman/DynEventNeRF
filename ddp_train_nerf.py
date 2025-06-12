#!/usr/bin/env python3
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.optim
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
from tensorboardX import SummaryWriter
from tqdm import trange

from ddp_config import config_parser, logger, setup_logger
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize
from data_loader_split import load_event_data_split
from nerf_sample_ray_split import CameraManager
from ddp_sampling import intersect_cylinder, intersect_sphere, perturb_samples, sample_pdf
from render_single_image import render_single_image
from create_nerf import create_nerf
from tonemapping import Gamma22, EventLogSpace


def log_view_to_tb(writer, global_step, log_data, gt_events, gt_rgb, mask, prefix=''):
    # rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    events_im = img_HWC2CHW(torch.from_numpy(gt_events))
    events_im = (events_im)/40+0.5
    writer.add_image(prefix + 'events_gt', events_im, global_step)

    rgb_im = img_HWC2CHW(Gamma22.from_linear(torch.from_numpy(gt_rgb)))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)

    for m in range(len(log_data)):
        # rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        # rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)

        rgb_im = img_HWC2CHW(Gamma22.from_linear(log_data[m]['rgb_linear']))
        writer.add_image(prefix + 'level_{}/rgb_norm'.format(m), (rgb_im-rgb_im.min(2, True)[0].min(1, True)[0])/(0.001+rgb_im.max(2,True)[0].max(1,True)[0]-rgb_im.min(2,True)[0].min(1,True)[0]), global_step)
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)

        rgb_im = img_HWC2CHW(Gamma22.from_linear(log_data[m]['fg_rgb_linear']))
        writer.add_image(prefix + 'level_{}/fg_rgb_norm'.format(m), (rgb_im-rgb_im.min(2, True)[0].min(1, True)[0])/(0.001+rgb_im.max(2,True)[0].max(1,True)[0]-rgb_im.min(2,True)[0].min(1,True)[0]), global_step)
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/fg_rgb'.format(m), rgb_im, global_step)

        depth = log_data[m]['fg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        writer.add_image(prefix + 'level_{}/fg_depth'.format(m), depth_im, global_step)

        if 'fg_ldist' in log_data[m]:
            ldist = log_data[m]['fg_ldist']
            ldist_im = img_HWC2CHW(colorize(ldist, cmap_name='jet', append_cbar=True,
                                            mask=mask))
            writer.add_image(prefix + 'level_{}/fg_ldist'.format(m), ldist_im, global_step)

        # rgb_im = img_HWC2CHW(log_data[m]['bg_rgb'])
        # rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/bg_rgb'.format(m), rgb_im, global_step)
        # depth = log_data[m]['bg_depth']
        # depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
        #                                 mask=mask))
        # writer.add_image(prefix + 'level_{}/bg_depth'.format(m), depth_im, global_step)
        bg_lambda = log_data[m]['bg_lambda']
        bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='hot', append_cbar=True,
                                            mask=mask))
        writer.add_image(prefix + 'level_{}/bg_lambda'.format(m), bg_lambda_im, global_step)


def get_sample_sizes(total, split_count):
    overhead = total % split_count
    normal_size = total // split_count
    sizes = [normal_size]*(split_count-overhead)+[normal_size+1]*overhead
    np.random.shuffle(sizes)
    assert np.sum(sizes) == total
    assert len(sizes) == split_count
    return sizes


def ddp_train_nerf(rank, args):
    ###### decide chunk size according to gpu memory
    logger.info('gpu_mem: {}'.format(torch.cuda.get_device_properties(rank).total_memory))
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    elif torch.cuda.get_device_properties(rank).total_memory / 1e9 > 7:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096
    else:
        logger.info('setting batch size according to 4G gpu')
        args.N_rand = 512//4
        args.chunk_size = 4096//4

    ###### Create log dir and copy the config file
    if rank == 0:
        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        f = os.path.join(args.basedir, args.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(args.basedir, args.expname, 'config.txt')
            with open(f, 'w') as out_file, open(args.config, 'r') as inp_file:
                out_file.write(inp_file.read())
    # torch.distributed.barrier()

    camera_mgr = CameraManager(learnable=False)
    ray_samplers = load_event_data_split(args.datadir, args.scene, camera_mgr=camera_mgr, split=args.train_split,
                                         skip=args.trainskip,
                                         use_ray_jitter=args.use_ray_jitter,
                                         polarity_offset=args.polarity_offset,
                                         damping_strength=args.damping_strength,
                                         tstart=args.tstart, tend=args.tend,
                                         is_rgb_only=args.is_rgb_only)
    # TODO: ray jitter should be off for the testing (if det is requested)
    # it is off anyway because validation uses render_single_image that ignores the jitter
    val_ray_samplers = load_event_data_split(args.datadir, args.scene, camera_mgr=camera_mgr, split='validation',
                                         skip=args.testskip,
                                         use_ray_jitter=args.use_ray_jitter,
                                         polarity_offset=args.polarity_offset,
                                         damping_strength=args.damping_strength,
                                         is_rgb_only=args.is_rgb_only)
                                        #  tstart=args.tstart, tend=args.tend)
                                        # todo


    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args, camera_mgr, use_lr_scheduler=args.use_lr_scheduler)
    # start, models = create_nerf(rank, args, camera_mgr, load_camera_mgr=False, load_optimizer=False)

    ##### important!!!
    # make sure different processes sample different rays
    np.random.seed((rank + 1) * 777+args.seed_offset)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((rank + 1) * 777+args.seed_offset)

    ##### only main process should do the logging
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.basedir, args.expname))

    scaler = GradScaler()
    # start training
    what_val_to_log = 0  # helper variable for parallel rendering of a image
    what_train_to_log = 0

    for i in trange(len(ray_samplers)):
        ray_samplers[i].update_rays(models['camera_mgr'])

    # for global_step in range(start + 1, start + 1 + args.N_iters):
    for global_step in range(start + 1, args.N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()
        ### Start of core optimization loop
        scalars_to_log['resolution'] = ray_samplers[0].resolution_level
        # randomly sample rays and move to device
        i = np.random.randint(low=0, high=len(ray_samplers))

        # current_neg_ratio = min(global_step/args.neg_ratio_anneal, 1.0) * args.neg_ratio
        current_neg_ratio = args.neg_ratio if global_step > args.neg_ratio_anneal else 0.0
        scalars_to_log['neg_ratio'] = current_neg_ratio

        ray_batches = dict()
        # potential problem: multiple reference frames
        sizes = get_sample_sizes(args.N_rand, len(ray_samplers))

        # todo: smart window size randomization
        # start_time = np.random.random()
        # end_time = np.random.random()
        # if end_time < start_time:
            # start_time, end_time = end_time, start_time
        # duration = np.random.random()*0.2+0.1
        # start_time = np.random.random()*(1-duration)
        # end_time = start_time+duration
        mid_time = np.random.random()
        duration = np.random.random()#*0.2+0.1
        start_time = max(0, mid_time-duration/2)
        end_time = min(1, mid_time+duration/2)

        for i in range(len(ray_samplers)):

            # todo: should pass start time and end time
            ray_batch = ray_samplers[i].random_sample(sizes[i], start_time, end_time, neg_ratio=current_neg_ratio)
            for key in ray_batch:
                # print(key, torch.is_tensor(ray_batch[key]), ray_batch[key])
                if torch.is_tensor(ray_batch[key]):
                    ray_batch[key] = ray_batch[key].to(rank)

                ray_batches.setdefault(key, []).append(ray_batch[key])

        del ray_batch
        ray_batch_combined = dict()

        for key in ray_batches:
            # print('key', key)
            if torch.is_tensor(ray_batches[key][0]):
                ray_batch_combined[key] = torch.concat(ray_batches[key], 0)
            elif all(x is None for x in ray_batches[key]):
                ray_batch_combined[key] = None
            else:
                # pass through the list
                ray_batch_combined[key] = ray_batches[key]

        # print(ray_batch_combined)

        # breakpoint()
        # 1/0

        ray_batch = ray_batch_combined
        # forward and backward
        # all_rets = []  # results on different cascade levels

        optim = models['optim']
        lr_scheduler = models['lr_scheduler']
        net = models['net']

        if args.optimize_transform: # todo: get rid of this transforms stuff?
            net.unfreeze_transform()

            # if there is init ckpt that we should've started with, preoptimize just the transform
            if global_step < args.N_iters_transform and args.init_ckpt_path:
                net.freeze_backend()  # optimize just the transform
            else:
                net.unfreeze_backend()
        else:
            net.freeze_transform()
            net.unfreeze_backend()

        optim.zero_grad()

        for m in range(models['cascade_level']):
            with autocast():
                # sample depths
                N_samples = models['cascade_samples'][m]
                if m == 0:
                    # foreground depth

                    ray_o = ray_batch['ray_o']
                    ray_d = ray_batch['ray_d']
                    fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                    fg_near_depth = ray_batch['min_depth']  # [..., ]

                    # ray_o, ray_d = bound_transform.inverse(ray_o, ray_d)

                    # ----
                    fg_near_depth_cyl, fg_far_depth_cyl = intersect_cylinder(ray_o, ray_d, args.crop_r, args.crop_y_min, args.crop_y_max)
                    assert fg_far_depth_cyl.shape == fg_far_depth_cyl.shape
                    assert fg_near_depth_cyl.shape == fg_near_depth_cyl.shape

                    fg_near_depth = torch.maximum(fg_near_depth, fg_near_depth_cyl)
                    fg_far_depth = torch.minimum(fg_far_depth, fg_far_depth_cyl)
                    # ----

                    step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                    fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                    fg_depth = perturb_samples(fg_depth)  # random perturbation during training

                    fg_depth_start = fg_depth
                    fg_depth_end = fg_depth
                    fg_depth_ref = fg_depth

                else:
                    if not args.is_rgb_only:
                        # sample pdf and concat with earlier samples
                        fg_weights = ret_start['fg_weights'].clone().detach()
                        fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                        fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                        fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                                      N_samples=N_samples, det=False)  # [..., N_samples]
                        fg_depth_start, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                        # todo: try accumulating jointly, not individually
                        fg_weights = ret_end['fg_weights'].clone().detach()
                        fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                        fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                        fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                                      N_samples=N_samples, det=False)  # [..., N_samples]
                        fg_depth_end, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                    # todo: try accumulating jointly, not individually
                    fg_weights = ret_ref['fg_weights'].clone().detach()
                    fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                    fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                    fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                                  N_samples=N_samples, det=False)  # [..., N_samples]
                    fg_depth_ref, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                    # sample pdf and concat with earlier samples
                    # bg_weights = ret['bg_weights'].clone().detach()
                    # bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                    # bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                    # bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                    #                               N_samples=N_samples, det=False)  # [..., N_samples]
                    # bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                all_rets = []
                if not args.is_rgb_only:
                    ret_start = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['start_ray_t'], fg_far_depth, fg_depth_start, ray_batch['background_linear'], global_step)
                    ret_end = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['end_ray_t'], fg_far_depth, fg_depth_end, ray_batch['background_linear'], global_step)
                    all_rets += [ret_start, ret_end]
                ret_ref = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['ref_ray_t'], fg_far_depth, fg_depth_ref, ray_batch['background_linear'], global_step)
                all_rets += [ret_ref]

                # all_rets.append(ret)

                if not args.is_rgb_only:
                    events_gt = ray_batch['events'].to(rank)
                # rgb_gt = ray_batch['rgb'].to(rank)
                color_mask = ray_batch['color_mask'].to(rank)

                mask_gt = ray_batch['mask'].to(rank) if ray_batch['mask'] is not None else None

                # event_mask = events_gt[..., 0] != 0
                # event_mask = None
                event_mask = mask_gt
                eps = args.tonemap_eps

                if not args.is_rgb_only:
                    start_log = EventLogSpace.from_linear(ret_start['rgb_linear'], eps)
                    end_log = EventLogSpace.from_linear(ret_end['rgb_linear'], eps)

                    diff = end_log - start_log

                    diff = diff * color_mask
                    events_gt = events_gt * color_mask

                    THR = args.event_threshold
                    event_loss = img2mse(diff, events_gt*THR, event_mask)
                    event_random_loss = img2mse(diff*0, events_gt*THR, event_mask)
                else:
                    event_loss = torch.zeros(1)
                    event_random_loss = torch.zeros(1)

                # assert mask_gt is None
                ref_rgb_gt_linear = ray_batch['rgb_linear'].to(rank)

                ref_rgb_render_linear = ret_ref['rgb_linear']

                if ref_rgb_gt_linear.shape[-1] == 1:
                    # we're dealing with color pre-filtered gt, so we need to apply Bayer mask onto the renderings
                    ref_rgb_render_linear = (ref_rgb_render_linear * color_mask).sum(-1, keepdim=True)

                assert ref_rgb_render_linear.shape == ref_rgb_gt_linear.shape
                assert mask_gt is None or mask_gt.shape == ref_rgb_gt_linear.shape[:-1], f'gt mask shape={mask_gt.shape}, ref shape={ref_rgb_gt_linear.shape}'
                ref_rgb_loss = img2mse(ref_rgb_render_linear, ref_rgb_gt_linear, mask=mask_gt)
                # ref_rgb_loss = 0.

                if not args.is_rgb_only:
                    ref_rgb_gt_log = EventLogSpace.from_linear(ref_rgb_gt_linear, eps)

                    events_from_ref_to_end_gt = ray_batch['events_from_ref_to_end']
                    end_rgb_gt_log = (ref_rgb_gt_log + events_from_ref_to_end_gt * THR)
                    ref_acc_loss = img2mse(end_log * color_mask, end_rgb_gt_log * color_mask, mask=mask_gt)
                else:
                    ref_acc_loss = torch.zeros(1)

                # --- ablation options ---
                if not args.use_event_loss:
                    event_loss = event_loss * 0
                if not args.use_rgb_loss:
                    ref_rgb_loss = ref_rgb_loss * 0
                if not args.use_accumulation_loss:
                    ref_acc_loss = ref_acc_loss * 0
                # --- ablation options end ---

                if not args.is_rgb_only:
                    loss = event_loss + ref_rgb_loss*0.01 + ref_acc_loss
                else:
                    loss = ref_rgb_loss
                # loss = rgb_loss
                # loss = img2mse(ret['rgb'], rgb_gt)
                # loss = img2mse(ret['rgb'], (diff_gt-diff_gt.min())/(diff_gt.max()-diff_gt.min()))

                if args.use_ldist_reg:
                    ldist = 0.
                    for ret in all_rets:
                    # for ret in [ret_start, ret_end]:
                        ldist = ldist + ret['fg_ldist'].mean()
                    loss = loss + ldist * args.ldist_reg

                if args.use_tv_reg:
                    tv = 0.
                    for ret in all_rets:
                    # for ret in [ret_start, ret_end]:
                        tv = tv + ret['fg_tv'].mean()
                    loss = loss + tv * args.tv_reg

                if args.use_tensorf_sparsity:
                    if global_step >= args.tensorf_sparsity_startit:
                        trf_reg = net.fg_net.get_sparsity_reg()  # todo: ugly
                        loss = loss + trf_reg * args.tensorf_sparsity
                    else:
                        trf_reg = loss * 0.0  # instead of 0.0, so that trf_reg.item() works later

                if args.use_tensorf_smoothness:
                    trf_smoothness = net.fg_net.get_smoothness_reg()
                    loss = loss + trf_smoothness * args.tensorf_smoothness

                if args.use_tensorf_tv:
                    trf_tv = net.fg_net.get_tv_reg()
                    loss = loss + trf_tv * args.tensorf_tv

                # sparsify as much as possible
                lambda_loss = 0.
                for ret in all_rets:
                # for ret in [ret_start, ret_end]:
                    bg_lambda = ret['bg_lambda'] #.mean()
                    lambda_loss = lambda_loss + (1-bg_lambda).mean() #todo: why is it mean() here as well?
                    # lambda_loss = lambda_loss + torch.log(torch.clamp(1-bg_lambda, min=1e-4)).mean()
                    # lambda_loss = lambda_loss + lambda_loss + torch.log(torch.clamp(bg_lambda, min=1e-4)).mean()
                    # lambda_loss = lambda_loss + ((1-bg_lambda)**2).mean()
                # lambda_reg_factor = 0. if global_step < 4000 else 1.
                lambda_reg_factor = 1-np.exp(-global_step/args.N_anneal_lambda)
                # lambda_reg_factor = 1.
                loss = loss + args.lambda_reg * lambda_reg_factor * lambda_loss

                # scalars_to_log['exposure_log'.format(m)] = net.exposure_log.item()
                scalars_to_log['level_{}/loss'.format(m)] = loss.item()
                scalars_to_log['level_{}/event_loss'.format(m)] = event_loss.item()
                scalars_to_log['level_{}/ref_rgb_loss'.format(m)] = ref_rgb_loss.item()
                scalars_to_log['level_{}/ref_acc_loss'.format(m)] = ref_acc_loss.item()
                scalars_to_log['level_{}/lambda_loss'.format(m)] = lambda_loss.item()
                scalars_to_log['lambda_reg_factor'.format(m)] = lambda_reg_factor
                if args.is_rgb_only:
                    scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(ref_rgb_loss.item())
                else:
                    scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(event_loss.item())
                # scalars_to_log['level_{}/pnsr_rgb'.format(m)] = mse2psnr(rgb_loss.item())
                # scalars_to_log['level_{}/range_loss'.format(m)] = range_loss.item()
                # scalars_to_log['level_{}/lambda_loss'.format(m)] = lambda_loss.item()
                scalars_to_log['level_{}/random_loss'.format(m)] = event_random_loss.item()
                # scalars_to_log['level_{}/real_div_random'.format(m)] = (event_loss.item()/max(0.01, event_random_loss.item()))
                # scalars_to_log['level_{}/real_minus_random'.format(m)] = (event_loss-event_random_loss).item()
                # scalars_to_log['level_{}/alpha_loss'.format(m)] = alpha_loss.item()
                if args.use_ldist_reg:
                    scalars_to_log['level_{}/ldist'.format(m)] = ldist.item()
                if args.use_tv_reg:
                    scalars_to_log['level_{}/tv'.format(m)] = tv.item()
                if args.use_tensorf_sparsity:
                    scalars_to_log['level_{}/trf_reg'.format(m)] = trf_reg.item()
                if args.use_tensorf_smoothness:
                    scalars_to_log['level_{}/trf_smooth'.format(m)] = trf_smoothness.item()
                if args.use_tensorf_tv:
                    scalars_to_log['level_{}/trf_tv'.format(m)] = trf_tv.item()

            scaler.scale(loss).backward()
            # for pgi, pg in enumerate(optim.param_groups):
            #     for pi, p in enumerate(pg['params']):
            #         scalars_to_log['level_{}_grad_norm/{}_{}'.format(m, pgi, pi)] = torch.mean(p.grad**2)**0.5

            # # clean unused memory
            # torch.cuda.empty_cache()

        scaler.step(optim)
        scaler.update()
        scalars_to_log['learning_rate'.format(m)] = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        ### only main process should do the logging
        if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
            logstr = '{} step: {} '.format(args.expname, global_step)
            for k in scalars_to_log:
                logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                writer.add_scalar(k, scalars_to_log[k], global_step)
            logger.info(logstr)

        ### each process should do this; but only main process merges the results
        if (global_step % args.i_img == 0 or global_step == start + 1) and (args.i_img > 0):
            #### critical: make sure each process is working on the same random image
            time0 = time.time()
            idx = what_val_to_log % len(val_ray_samplers)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
                # with record_function("render_single_image_val"):
            log_data = render_single_image(rank, args.world_size, models, val_ray_samplers[idx], args.chunk_size,
                                           iteration=global_step, args=args, timestamp=end_time)
            # prof.export_chrome_trace("render.json")
            # prof.export_stacks("render_stacks_cpu", metric='self_cpu_time_total')
            # prof.export_stacks("render_stacks_cuda", metric='self_cuda_time_total')
            what_val_to_log += 1
            dt = time.time() - time0
            if rank == 0:  # only main process should do this
                logger.info('Logged a random validation view in {} seconds, t={}'.format(dt, end_time))
                log_view_to_tb(writer, global_step, log_data,
                               gt_events=val_ray_samplers[idx].get_img(start_time, end_time),
                               gt_rgb=val_ray_samplers[idx].get_rgb(end_time),
                               mask=None,
                               prefix='val/')

            time0 = time.time()
            idx = what_train_to_log % len(ray_samplers)
            log_data = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size,
                                           global_step, args, timestamp=start_time)
            what_train_to_log += 1
            dt = time.time() - time0
            if rank == 0:  # only main process should do this
                logger.info('Logged a random training view in {} seconds, t={}'.format(dt, start_time))
                log_view_to_tb(writer, global_step, log_data,
                               gt_events=ray_samplers[idx].get_img(start_time, end_time),
                               gt_rgb=ray_samplers[idx].get_rgb(start_time),
                               mask=None,
                               prefix='train/')
                writer.flush()

            del log_data
            torch.cuda.empty_cache()

        if rank == 0 and (global_step % args.i_weights == 0 and global_step > 0):
            # saving checkpoints and logging
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            to_save = OrderedDict()

            name = 'net'
            to_save[name] = models[name].state_dict()

            name = 'optim'
            to_save[name] = models[name].state_dict()

            name = 'lr_scheduler'
            to_save[name] = models[name].state_dict()

            name = 'camera_mgr'
            to_save[name] = models[name].state_dict()

            torch.save(to_save, fpath)


def train():
    parser = config_parser()
    args = parser.parse_args()
    if 'SLURM_JOB_ID' in os.environ:
        args.slurmjob = os.environ['SLURM_JOB_ID']
    logger.info(parser.format_values())
    args.world_size = 1

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    #     try:
    #         ddp_train_nerf(0, args)
    #     except KeyboardInterrupt:
    #         logger.warning('Keyboard interrupt received, shutting down...')
    #         pass
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")
    # prof.export_stacks("profiler_stacks_cuda.txt", "self_cuda_time_total")
    # prof.export_stacks("profiler_stacks_cpu.txt", "self_cpu_time_total")
    ddp_train_nerf(0, args)




if __name__ == '__main__':
    setup_logger()
    train()
