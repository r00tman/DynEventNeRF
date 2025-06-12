from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import cv2
import imageio
import os
from os import path
from tqdm import trange
from functools import lru_cache
# import numba

from ddp_config import logger
from distortion import undistort_norm
from edi_cpp import EventStorage
from tonemapping import Gamma22


########################################################################################################################
# ray batch sampling
########################################################################################################################

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################

class DistortionCache:
    def __init__(self):
        self.cache = dict()

    def get_uv_by_hash(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        fn = path.join('.dist_cache', f'{idx}.npz')
        logger.warning(fn)
        if path.exists(fn):
            logger.warning('exists')
            uv = np.load(fn)
            u, v = uv['u'], uv['v']
            self.cache[idx] = (u, v)
            return u, v
        else:
            logger.warning('doesnt exist')
            return None, None

    def get_uv(self, H, W, intrinsics):
        idx = hash((H, W, *intrinsics.reshape(-1).tolist()))
        return self.get_uv_by_hash(idx)

    def store_uv(self, H, W, intrinsics, u, v):
        idx = hash((H, W, *intrinsics.reshape(-1).tolist()))
        fn = path.join('.dist_cache', f'{idx}.npz')
        os.makedirs('.dist_cache', exist_ok=True)
        np.savez(fn, u=u, v=v)
        self.cache[idx] = (u, v)

dcache = DistortionCache()
import time

def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''

    # tstart = time.time()

    u, v = dcache.get_uv(H, W, intrinsics)
    if u is None:
        logger.warning('recomputing')
        # c2w = torch.from_numpy(c2w)

        u, v = torch.meshgrid(torch.arange(W, device=c2w.device), torch.arange(H, device=c2w.device))
        u, v = u.T, v.T

        u = u.reshape(-1).float() + 0.5    # add half pixel
        v = v.reshape(-1).float() + 0.5

        # ---- distort rays to match the camera ----
        # todo: this is only an approximation
        # the correct way to do it is to use non-linear solver and
        # precompute it for each camera
        # intrinsics = intrinsics.cpu().numpy()
        f = [intrinsics[0,0], intrinsics[1,1]]
        c = [intrinsics[0,2], intrinsics[1,2]]
        # k = [intrinsics[4,0], intrinsics[4,1]]
        distortion = intrinsics[4]
        # logger.info(f'{f} {c} {k} {u.max()} {v.max()}')

        xs = ((u-c[0])/f[0]).cpu().numpy()
        ys = ((v-c[1])/f[1]).cpu().numpy()

        xsnew = xs[:]
        ysnew = ys[:]

        for i in trange(len(xs)):
            x, y = xs[i], ys[i]
            xnew, ynew = undistort_norm(x, y, distortion)
            xsnew[i], ysnew[i] = xnew, ynew

        # r2 = x**2+y**2
        # dist =  1+k[0]*r2+k[1]*r2*r2
        # x = x/dist
        # y = y/dist

        u = xsnew*f[0]+c[0]
        v = ysnew*f[1]+c[1]
        # logger.info(f'{u.min()} {v.min()} {u.max()} {v.max()} {x.min()} {x.max()} {y.min()} {y.max()}')
        dcache.store_uv(H, W, intrinsics, u, v)
    else:
        pass
    intrinsics = torch.from_numpy(intrinsics).to(c2w.device)
    u = torch.tensor(u).to(c2w.device)
    v = torch.tensor(v).to(c2w.device)
    # ------------------------------------------


    pixels = torch.stack((u, v, torch.ones_like(u)), axis=0)  # (3, H*W)

    ray_matrix = torch.matmul(c2w[:3, :3], torch.inverse(intrinsics[:3, :3]))
    # rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    # rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = torch.matmul(ray_matrix, pixels)  # (3, H*W)
    rays_d = rays_d.transpose(1, 0)  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = torch.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = torch.inverse(c2w)[2, 3]
    depth = depth * torch.ones((rays_o.shape[0],), dtype=rays_o.dtype, device=rays_o.device)  # (H*W,)

    # rays_o = rays_o.cpu().numpy()
    # rays_d = rays_d.cpu().numpy()
    # depth = depth.cpu().numpy()
    # ray_matrix = ray_matrix.cpu().numpy()
    # tend = time.time()
    # logger.warning(f'get rays: {(tend-tstart)*1000}ms')

    return rays_o, rays_d, depth, ray_matrix


class CameraManager(nn.Module):
    def __init__(self, learnable=False):
        super().__init__()
        self.learnable = learnable
        self.c2w_store = nn.ParameterDict()

    def encode_name(self, name):
        return name.replace('.', '-')

    def add_camera(self, name, c2w):
        key = self.encode_name(name)
        self.c2w_store[key] = nn.Parameter(torch.from_numpy(c2w))

    def contains(self, name):
        key = self.encode_name(name)
        return key in self.c2w_store

    def get_c2w(self, name):
        key = self.encode_name(name)
        res = self.c2w_store[key]
        if not self.learnable:
            res = res.detach()
        return res



# @numba.jit()
# def accumulate_events(xs, ys, ts, ps, out, resolution_level, polarity_offset):
#     for i in range(len(xs)):
#         # x, y, t, p = xs[i], ys[i], ts[i], ps[i]
#         x, y, p = xs[i], ys[i], ps[i]
#         out[y // resolution_level, x // resolution_level] += p+polarity_offset


class RaySamplerSingleEventStream:
    def __init__(self, H, W, intrinsics,
                       events=None,
                       rgb_paths=None,
                       mask_path=None,
                       background_path=None,
                       resolution_level=1,
                       use_ray_jitter=True,
                       polarity_offset=0.0,
                       damping_strength=0.93,
                       tstart=0.,
                       tend=1000.,
                       is_rgb_only=False):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics

        self.is_rgb_only = is_rgb_only
        self.events = events
        self.polarity_offset = polarity_offset
        self.damping_strength = damping_strength
        self.tstart = tstart
        self.tend = tend

        self.mask_path = mask_path
        self.background_path = background_path
        self.rgb_paths = rgb_paths
        self.view_name = rgb_paths[0][1]  # take the first frame as name

        self.rgb_linear = None
        self.mask = None
        self.background_linear = None

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

        xs, ys, ts, ps = self.events
        xs = torch.tensor(xs, dtype=torch.int16)
        ys = torch.tensor(ys, dtype=torch.int16)
        ts = torch.tensor(ts, dtype=torch.float32)
        ps = torch.tensor(ps, dtype=torch.int8)
        self.event_storage = EventStorage(H, W, damping_strength, xs, ys, ts, ps)

        self.use_ray_jitter = use_ray_jitter

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level


            self.color_mask = np.zeros((self.H, self.W, 3))

            self.color_mask[0::2, 0::2, 0] = 1  # r

            self.color_mask[0::2, 1::2, 1] = 1  # g
            self.color_mask[1::2, 0::2, 1] = 1  # g

            self.color_mask[1::2, 1::2, 2] = 1  # b

            self.color_mask = self.color_mask.reshape((-1, 3))

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                if len(self.mask.shape) == 3:  # if mask is not monochrome (e.g., RGB/RGBA), take R
                    logger.warning(f'mask shape {self.mask.shape} - taking first channel only')
                    self.mask = self.mask[..., 0]
                self.mask = self.mask.reshape((-1,))
            else:
                self.mask = None

            if self.background_path is not None:
                self.background_linear = imageio.imread(self.background_path).astype(np.float32) / 255.
                self.background_linear = cv2.resize(self.background_linear, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                assert len(self.background_linear.shape) == 3  # only RGB background counts
                logger.warning(f'background shape {self.background_linear.shape}')
                if self.background_linear.shape[2] == 4:
                    self.background_linear = self.background_linear[..., :3]
                self.background_linear = self.background_linear.reshape((-1, 3))
                self.background_linear = Gamma22.to_linear(self.background_linear)
            else:
                self.background_linear = None

            if self.rgb_paths is not None:
                logger.info(f'rgbs: {self.rgb_paths}')

                self.rgbs_linear = []
                for frame_number, path in self.rgb_paths:
                    rgb_linear = imageio.imread(path).astype(np.float32) / 255.
                    rgb_linear = cv2.resize(rgb_linear, (self.W, self.H), interpolation=cv2.INTER_AREA)

                    if len(rgb_linear.shape) == 2:
                        # # todo: this needs to go. there's no good reason for that anymore
                        # #       it only makes things worse with the current pre-masked rgb input
                        # #       no way to detect if they are 3 channels too
                        # rgb = np.tile(rgb[..., None], (1, 1, 3)).reshape((-1, 3))
                        rgb_linear = rgb_linear[..., None].reshape((-1, 1))
                    elif rgb_linear.shape[2] == 4:
                        # mask = rgb[..., 3].reshape((-1,))
                        rgb_linear = rgb_linear[..., :3].reshape((-1, 3))
                    elif rgb_linear.shape[2] == 3:
                        rgb_linear = rgb_linear.reshape((-1, 3))
                    rgb_linear = Gamma22.to_linear(rgb_linear)
                    self.rgbs_linear.append((frame_number, rgb_linear))
            else:
                self.rgbs_linear = None

        self.rays_o, self.rays_d, self.depth, self.ray_matrix = None, None, None, None

    def update_rays(self, camera_mgr):
        c2w_mat = camera_mgr.get_c2w(self.view_name)

        self.rays_o, self.rays_d, self.depth, self.ray_matrix = \
                get_rays_single_image(self.H, self.W, self.intrinsics, c2w_mat)

    def get_img(self, start_t, end_t):
        if self.event_storage is not None:
            event_frame = self.event_storage.accumulate(self.map_time(start_t), self.map_time(end_t))
            event_frame = event_frame.numpy()

            # (...,) to (..., 3)
            event_frame = np.tile(event_frame[..., None], (1, 1, 3))
            event_frame = event_frame.reshape((self.H, self.W, 3))
            return event_frame
        else:
            return None

    def get_closest_rgb_linear(self, timestamp):
        dists = [(abs(self.reverse_map_time(frame_number) - timestamp), image_idx)
                 for image_idx, (frame_number, _) in enumerate(self.rgbs_linear)]

        min_dist, min_idx = min(dists)
        closest_rgb_linear = self.rgbs_linear[min_idx][1]
        return closest_rgb_linear

    def get_rgb(self, timestamp=0):
        if self.rgbs_linear is not None:
            # return self.rgb.reshape((self.H, self.W, 3))
            # take the first one
            closest_rgb_linear = self.get_closest_rgb_linear(timestamp)
            # return self.rgbs_linear[0][1].reshape((self.H, self.W, -1))
            return closest_rgb_linear.reshape((self.H, self.W, -1))
        else:
            return None

    def get_all(self, timestamp):
        # todo: time
        min_depth = 1e-4 * torch.ones_like(self.rays_d[..., 0])
        rays_t = timestamp * torch.ones_like(self.rays_d[..., 0])

        closest_rgb_linear = self.get_closest_rgb_linear(timestamp)

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('ray_t', rays_t),
            ('depth', self.depth),
            ('min_depth', min_depth),

            # take the closest frame
            ('rgb_linear', closest_rgb_linear),
            ('color_mask', self.color_mask),
            ('mask', self.mask),
            ('background_linear', self.background_linear),
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def map_time(self, time):
        # maps [0,1] to [tstart, tend] frame number
        return time * (self.tend - self.tstart) + self.tstart

    def reverse_map_time(self, frame_number):
        # maps [tstart, tend] frame number to [0,1]
        return (frame_number - self.tstart) / (self.tend - self.tstart)

    def random_sample(self, N_rand, start_t, end_t, neg_ratio=0):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''

        # import time
        # tstart = time.time()

        event_frame = self.event_storage.accumulate(self.map_time(start_t), self.map_time(end_t))
        event_frame = event_frame.numpy()

        # todo: debug here
        dists = [(abs(self.reverse_map_time(num)-end_t), idx) for idx, (num, _) in enumerate(self.rgbs_linear)]
        dists.sort()
        # take the two closest: a < map_time(end_t) < b
        dists = dists[:2]

        ref_idx = dists[np.random.randint(len(dists))][1]
        ref_frame_number, ref_rgb_linear = self.rgbs_linear[ref_idx]
        ref_t = self.reverse_map_time(ref_frame_number)

        # ref_idx = np.random.randint(len(self.rgbs_linear))
        # ref_frame_number, ref_rgb_linear = self.rgbs_linear[ref_idx]
        # ref_t = self.reverse_map_time(ref_frame_number)

        if ref_t < end_t:
            # ref_t < end_t: [ref_t, end_t]
            event_frame_from_ref_to_end = self.event_storage.accumulate(self.map_time(ref_t), self.map_time(end_t))
        else:
            # ref_t >= end_t: -[end_t, ref_t]
            event_frame_from_ref_to_end = -self.event_storage.accumulate(self.map_time(end_t), self.map_time(ref_t))
        event_frame_from_ref_to_end = event_frame_from_ref_to_end.numpy()

        # (...,) to (..., 3)
        event_frame = np.tile(event_frame[..., None], (1, 1, 3))
        event_frame = event_frame.reshape((-1, 3))

        event_frame_from_ref_to_end = np.tile(event_frame_from_ref_to_end[..., None], (1, 1, 3))
        event_frame_from_ref_to_end = event_frame_from_ref_to_end.reshape((-1, 3))

        mask = np.nonzero(event_frame[..., 0])
        # print(mask)
        assert len(mask) == 1
        mask = mask[0]

        if mask.shape[0] > 0 and not self.is_rgb_only:
            # Random from one image
            # select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)
            pos_size = int(N_rand*(1-neg_ratio))
            pos_should_replace = pos_size>mask.shape[0]
            if pos_should_replace:
                logger.warning('sampling views with replacement (not enough events this frame)')
            select_inds_raw = np.random.choice(mask.shape[0], size=(pos_size,), replace=pos_should_replace)

            select_inds = mask[select_inds_raw]

            neg_inds = np.random.choice(self.H*self.W, size=(N_rand-select_inds_raw.shape[0],), replace=False)
            select_inds = np.concatenate([select_inds, neg_inds])
        else:
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)
            if not self.is_rgb_only:
                logger.warning('no events this frame, bad sampling')

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]
        ray_matrix = self.ray_matrix

        if self.use_ray_jitter:
            noise = torch.rand(2, len(select_inds), dtype=ray_matrix.dtype, device=ray_matrix.device)-0.5  # [2, N_rand]
            # noise = np.random.rand(2, len(select_inds)).astype(np.float32)-0.5  # [2, N_rand]
            noise = torch.stack((noise[0], noise[1], noise.new_zeros(len(select_inds))), axis=0)  # [3, N_rand]
            # noise = np.stack((noise[0], noise[1], np.zeros(len(select_inds), dtype=np.float32)), axis=0)  # [3, N_rand]
            # ornoise = noise
            noise = ray_matrix @ noise
            # noise = np.dot(ray_matrix.cpu().numpy(), noise)
            noise = noise.T
            # noise = noise.transpose((1, 0))  # [N_rand, 3]
            # noise = torch.from_numpy(noise).to(rays_d.device)

            assert noise.shape == rays_d.shape
            rays_d = rays_d + noise

        if self.events is not None:
            events = event_frame[select_inds, :]          # [N_rand, 3]
            events_from_ref_to_end = event_frame_from_ref_to_end[select_inds, :]
        else:
            events = None
            events_from_ref_to_end = None

        if self.rgbs_linear is not None:
            ref_rgb_linear = ref_rgb_linear[select_inds, :]          # [N_rand, 3], or [N_rand, 1]
        else:
            ref_rgb_linear = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.background_linear is not None:
            background_linear = self.background_linear[select_inds]
        else:
            background_linear = None

        min_depth = 1e-4 * torch.ones_like(rays_d[..., 0])

        start_rays_t = start_t * torch.ones_like(rays_d[..., 0])
        end_rays_t = end_t * torch.ones_like(rays_d[..., 0])
        ref_rays_t = ref_t * torch.ones_like(rays_d[..., 0])

        color_mask = self.color_mask[select_inds, :]

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),

            ('end_ray_t', end_rays_t),
            ('start_ray_t', start_rays_t),
            ('ref_ray_t', ref_rays_t),

            ('depth', depth),
            ('events', events),
            ('events_from_ref_to_end', events_from_ref_to_end),
            ('min_depth', min_depth),

            ('rgb_linear', ref_rgb_linear),
            ('color_mask', color_mask),
            ('mask', mask),
            ('background_linear', background_linear),
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret
