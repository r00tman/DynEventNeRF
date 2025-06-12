#!/usr/bin/env python3
import os
import time

import torch
import numpy as np
import ffmpeg
import imageio

from data_loader_split import load_event_data_split
from utils import mse2psnr, colorize_np, to8b
from ddp_config import setup_logger, logger, config_parser
from render_single_image import render_single_image
from create_nerf import create_nerf
from nerf_sample_ray_split import CameraManager
from tonemapping import Gamma22


class SequenceWriter:
    def __init__(self, out_dir, family, write_video):
        self.out_dir = out_dir
        self.family = family
        self.write_video = write_video
        self.video_stream = None
        self.width = None
        self.height = None

    def write(self, image, image_number):
        # todo: make it so that this is not needed
        if image_number.lower().endswith('.png') or image_number.lower().endswith('.jpg'):
            pass
        else:
            image_number = image_number + '.png'

        def prepend_family(fn):
            if self.family:
                fn = self.family + '_' + fn
            return fn

        def append_family(fn):
            if self.family:
                fn = fn + '_' + self.family
            return fn

        image = to8b(image)

        outfn = os.path.join(self.out_dir, prepend_family(image_number))
        imageio.imwrite(outfn, image)

        if self.write_video:
            # open the stream
            if self.video_stream is None:
                self.height, self.width = image.shape[:2]
                out_filename = append_family(self.out_dir.rstrip('/'))+'.mp4'
                self.video_stream = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(self.width, self.height), r=30)
                    .output(out_filename, pix_fmt='yuv444p', crf=10, blocksize=2048, flush_packets=1)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )

            assert image.shape == (self.height, self.width, 3)
            self.video_stream.stdin.write(
                image
                .astype(np.uint8)
                .tobytes()
            )


    def close(self):
        if self.video_stream is not None:
            self.video_stream.stdin.close()
            self.video_stream.wait()
            self.video_stream = None


def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    # setup(rank, args.world_size)
    ###### set up logger
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    camera_mgr = CameraManager(learnable=False)
    start, models = create_nerf(rank, args, camera_mgr, load_camera_mgr=False, load_optimizer=False)


    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        str_bullet_time = '_bt' if args.render_bullet_time else ''
        out_dir = os.path.join(args.basedir, args.expname,
                               f'render_{split}{str_bullet_time}_{start:06d}')
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)


        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_event_data_split(args.datadir, args.scene, split,
                view_filter=args.render_view,
                camera_mgr=models['camera_mgr'],
                use_ray_jitter=args.use_ray_jitter,
                polarity_offset=args.polarity_offset,
                skip=args.testskip)
        writers_by_family = dict()
        for idx in range(len(ray_samplers)):
            viewname = ''
            def write_image(image, family, fname):
                writer = writers_by_family.setdefault(
                    family,
                    SequenceWriter(out_dir, f'{viewname}_{family}', args.write_video))
                writer.write(image, fname)

            if args.render_bullet_time:
                # frame_indices = np.array([idx], dtype=np.int64)
                frame_indices = np.array([idx], dtype=np.int64)
                timestamps = (frame_indices / (len(ray_samplers)-1) * args.render_timestamp_periods)
            else:
                frame_indices = np.arange(args.render_timestamp_frames, dtype=np.int64)
                timestamps = (frame_indices / args.render_timestamp_frames * args.render_timestamp_periods)

            # timestamps = (timestamps * 1000).astype(np.int64)

            for frame_idx, timestamp in zip(frame_indices, timestamps):
                # timestamp = frame_idx / args.render_timestamp_frames * args.render_timestamp_periods
                # timestamp = int(timestamp * 1000)

                ### each process should do this; but only main process merges the results
                fname = '{:06d}.png'.format(idx)
                if ray_samplers[idx].view_name is not None:
                    fname = os.path.basename(ray_samplers[idx].view_name)
                    if '.' not in fname:
                        fname = os.path.basename(ray_samplers[idx].view_name)
                        # fname = fname+'.png'
                a, b = os.path.splitext(fname)
                ts_absolute = int(np.round(timestamp *(args.tend-args.tstart)+args.tstart))
                if ts_absolute < args.render_tstart or ts_absolute > args.render_tend:
                    logger.info('Skipping {} as {} is not in [{}, {}]'.format(fname, ts_absolute, args.render_tstart, args.render_tend))
                    continue
                timestamp = (ts_absolute-args.tstart)/(args.tend-args.tstart)
                # fname = f'{timestamp:04d}{b}'
                fname = f'{ts_absolute:04d}{b}'
                viewname = a

                if os.path.isfile(os.path.join(out_dir, fname)):
                    logger.info('Skipping {}'.format(fname))
                    continue

                time0 = time.time()
                # ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size, start, args, timestamp=timestamp/1000)
                ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size, start, args, timestamp=timestamp)
                dt = time.time() - time0
                if rank == 0:    # only main process should do this
                    # logger.info('Rendered {} in {} seconds at t={}'.format(fname, dt, timestamp/1000))
                    logger.info('Rendered {} in {} seconds at t={}'.format(fname, dt, timestamp))

                    # only save last level
                    im = Gamma22.from_linear(ret[-1]['rgb_linear']).numpy()
                    # compute psnr if ground-truth is available
                    # if ray_samplers[idx].view_name is not None:
                    #     gt_im = ray_samplers[idx].get_img()
                    #     psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                    #     logger.info('{}: psnr={}'.format(fname, psnr))

                    write_image(im, '', fname)

                    im = Gamma22.from_linear(ret[-1]['fg_rgb_linear']).numpy()
                    write_image(im, 'fg', fname)

                    # im = ret[-1]['bg_rgb'].numpy()
                    # im = to8b(im)
                    # imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)

                    im = ret[-1]['fg_depth'].numpy()
                    # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    # scene radius 1 means that the range is [0, 2].
                    # so if we need constant range, e.g., for animation purposes, this is it
                    im = colorize_np(im, cmap_name='jet', append_cbar=True, vmin=0.0, vmax=2.0)
                    write_image(im, 'fg_depth', fname)

                    # im = ret[-1]['bg_depth'].numpy()
                    # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    # im = to8b(im)
                    # imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

                    if 'fg_ldist' in ret[-1]:
                        im = ret[-1]['fg_ldist'].numpy()
                        im = colorize_np(im, cmap_name='jet', append_cbar=True)
                        write_image(im, 'fg_ldist', fname)

            torch.cuda.empty_cache()

            if not args.render_bullet_time:
                # clean up
                for writer in writers_by_family.values():
                    writer.close()
                    del writer
                writers_by_family = dict()

        # clean up
        for writer in writers_by_family.values():
            writer.close()
            del writer
        writers_by_family = dict()



    # clean up for multi-processing
    # cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.render_tstart < 0 :
        args.render_tstart = args.tstart

    if args.render_tend < 0 :
        args.render_tend = args.tend

    args.world_size = 1
    # if args.world_size == -1:
    #     args.world_size = torch.cuda.device_count()
    #     logger.info('Using # gpus: {}'.format(args.world_size))
    ddp_test_nerf(0, args)
    # torch.multiprocessing.spawn(ddp_test_nerf,
    #                             args=(args,),
    #                             nprocs=args.world_size,
    #                             join=True)


if __name__ == '__main__':
    # setup_logger()
    test()
