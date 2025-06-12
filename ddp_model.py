from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
# import numpy as np
from ddp_config import logger
from utils import TINY_NUMBER
from nerf_network import DummyEmbedder, Embedder, TimeAwareEmbedder, MLPNet, TCNNNet, TensoRFVMNet, TensoRFCPNet
from tonemapping import Gamma22


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


class BoundTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.translation = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        assert x.shape[-1] == 3
        x = x + self.translation
        return x


class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.bound_transform = BoundTransform()
        # self.exposure_log = nn.Parameter(torch.zeros(1))
        if args.use_pe:
            self.fg_embedder_position = TimeAwareEmbedder(input_dim=4,
                                                          max_freq_log2_pos=args.max_freq_log2_pos - 1,
                                                          N_freqs_pos=args.max_freq_log2_pos,
                                                          max_freq_log2_time=args.max_freq_log2_time - 1,
                                                          N_freqs_time=args.max_freq_log2_time,
                                                          N_anneal=args.N_anneal,
                                                          N_anneal_min_freq=args.N_anneal_min_freq,
                                                          use_annealing=args.use_annealing)
            self.fg_embedder_viewdir = Embedder(input_dim=3,
                                                max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                                N_freqs=args.max_freq_log2_viewdirs,
                                                N_anneal=args.N_anneal,
                                                N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
                                                use_annealing=args.use_annealing)
        else:
            self.fg_embedder_position = DummyEmbedder(input_dim=4)
            self.fg_embedder_viewdir = DummyEmbedder(input_dim=3)

        if args.backend == 'mlp':
            self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                                 input_ch=self.fg_embedder_position.out_dim,
                                 input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                                 use_viewdirs=args.use_viewdirs,
                                 act=args.activation,
                                 garf_sigma=args.garf_sigma,
                                 crop_y=(args.crop_y_min, args.crop_y_max),
                                 crop_r=args.crop_r,
                                 init_gain=args.init_gain)
        elif args.backend == 'tcnn':
            self.fg_net = TCNNNet(use_viewdirs=args.use_viewdirs,
                                  crop_y=(args.crop_y_min, args.crop_y_max),
                                  crop_r=args.crop_r,
                                  init_gain=args.init_gain)
        elif args.backend == 'tensorfvm':
            self.fg_net = TensoRFVMNet(N=args.tensorf_grid_dim, Ntime=args.tensorf_grid_dim_time,
                                       R=args.tensorf_rank,
                                       use_viewdirs=args.use_viewdirs,
                                       crop_y=(args.crop_y_min, args.crop_y_max),
                                       crop_r=args.crop_r,
                                       init_gain=args.init_gain)
        elif args.backend == 'tensorfcp':
            self.fg_net = TensoRFCPNet(N=args.tensorf_grid_dim, Ntime=args.tensorf_grid_dim_time,
                                       R=args.tensorf_rank,
                                       Hsteps=args.tensorf_Hsteps,
                                       Hmin=args.tensorf_Hmin, Hmin_time=args.tensorf_Hmin_time,
                                       Hiters=args.tensorf_Hiters,
                                       use_viewdirs=args.use_viewdirs,
                                       crop_y=(args.crop_y_min, args.crop_y_max),
                                       crop_r=args.crop_r,
                                       init_gain=args.init_gain)
        else:
            raise RuntimeError(f'invalid backend: {args.backend}')

        # # background; bg_pt is (x, y, z, 1/r)
        # self.bg_embedder_position = Embedder(input_dim=4,
        #                                      max_freq_log2=args.max_freq_log2 - 1,
        #                                      N_freqs=args.max_freq_log2,
        #                                      N_anneal=args.N_anneal,
        #                                      N_anneal_min_freq=args.N_anneal_min_freq,
        #                                      use_annealing=args.use_annealing)
        # self.bg_embedder_viewdir = Embedder(input_dim=3,
        #                                     max_freq_log2=args.max_freq_log2_viewdirs - 1,
        #                                     N_freqs=args.max_freq_log2_viewdirs,
        #                                     N_anneal=args.N_anneal,
        #                                     N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
        #                                     use_annealing=args.use_annealing)
        # self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
        #                      input_ch=self.bg_embedder_position.out_dim,
        #                      input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
        #                      use_viewdirs=args.use_viewdirs,
        #                      act=args.activation)

        self.with_bg = False
        self.with_ldist = args.use_ldist_reg
        self.with_tv = args.use_tv_reg

        self.bg_color = Gamma22.to_linear(args.bg_color/255.)

    def freeze_backend(self):
        for p in self.fg_net.parameters():
            p.requires_grad_(False)

    def unfreeze_backend(self):
        for p in self.fg_net.parameters():
            p.requires_grad_(True)

    def freeze_transform(self):
        for p in self.bound_transform.parameters():
            p.requires_grad_(False)

    def unfreeze_transform(self):
        for p in self.bound_transform.parameters():
            p.requires_grad_(True)


    def forward(self, ray_o, ray_d, ray_t, fg_z_max, fg_z_vals, bg_rgb_linear, iteration):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals: [..., N_samples]
        :return
        '''
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm  # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_t = ray_t.unsqueeze(-1).unsqueeze(-2).expand(dots_sh + [N_samples, 1])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        fg_pts = self.bound_transform(fg_pts)
        fg_pts = torch.cat([fg_pts, fg_ray_t], dim=-1)
        # input = torch.cat((self.fg_embedder_position(fg_pts, iteration),
        #                    self.fg_embedder_viewdir(fg_viewdirs, iteration)), dim=-1)
        # fg_raw = self.fg_net(input)
        # do the magic translation/rotation of fg_pts
        fg_raw = self.fg_net(fg_pts, fg_viewdirs, iteration=iteration,
                             embedder_position=self.fg_embedder_position,
                             embedder_viewdir=self.fg_embedder_viewdir)
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        # fg_rgb_map = torch.clamp_max(fg_rgb_map * F.softplus(self.exposure_log), 1.0)
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]


        fg_midpoint = (fg_z_vals[..., 1:] + fg_z_vals[..., :-1])/2
        fg_midpoint = ray_d_norm * torch.cat((fg_midpoint, (fg_z_max.unsqueeze(-1) + fg_z_vals[..., -1:])/2),
                                          dim=-1)

        fg_midpointdist = abs(fg_midpoint.unsqueeze(-1) - fg_midpoint.unsqueeze(-2))

        if self.with_ldist:
            fg_ldist1 = torch.sum(fg_weights.unsqueeze(-1)*fg_weights.unsqueeze(-2)*fg_midpointdist, (-2, -1))
            fg_ldist2 = torch.sum(1/3*(fg_weights**2)*fg_dists, -1)
            fg_ldist = fg_ldist1+fg_ldist2


        if self.with_tv:
            fg_tv = torch.sum(abs(fg_weights[..., 1:]-fg_weights[..., :-1]), -1)

        # # render background
        # N_samples = bg_z_vals.shape[-1]
        # bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        # bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        # bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        # bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        # input = torch.cat((self.bg_embedder_position(bg_pts, iteration),
        #                    self.bg_embedder_viewdir(bg_viewdirs, iteration)), dim=-1)
        # # near_depth: physical far; far_depth: physical near
        # input = torch.flip(input, dims=[-2, ])
        # bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
        # bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        # bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        # bg_raw = self.bg_net(input)
        # bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # # Eq. (3): T
        # # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        # T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        # T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        # bg_weights = bg_alpha * T  # [..., N_samples]

        # bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        # bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # # composite foreground and background
        # bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        # bg_depth_map = bg_lambda * bg_depth_map
        if self.with_bg:
            assert False
            bg_rgb_map = 0 # todo: fix
            rgb_map = fg_rgb_map + bg_rgb_map
        elif bg_rgb_linear is not None:
            # we expect bg_rgb to be already in linear space, not sRGB, as loss will convert it back?
            # let's check
            # nevermind, we're in srgb all the time because of the rgb-only mode
            # so we expect it
            # wait, we need to debayer the background, as it is in grayscale. dammit
            # ok, then let's use debayered background, as it is sharper
            # and maybe even debayered ground-truth too
            rgb_map = fg_rgb_map + bg_lambda.unsqueeze(-1)*bg_rgb_linear
        else:
            # rgb_map = fg_rgb_map + bg_lambda.unsqueeze(-1)*(159./255.)  # hard coded value of background in sRGB = 159/255
            rgb_map = fg_rgb_map + bg_lambda.unsqueeze(-1)*self.bg_color  # hard coded value of background in sRGB = 159/255

        ret = OrderedDict([('rgb_linear', rgb_map),            # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           # ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb_linear', fg_rgb_map.detach()),      # below are for logging
                           ('fg_depth', fg_depth_map.detach()),
                           # ('bg_rgb', bg_rgb_map.detach()),
                           # ('bg_depth', bg_depth_map.detach()),
                           ('bg_lambda', bg_lambda)
                           ])
        if self.with_ldist:
            ret['fg_ldist'] = fg_ldist      # distortion regularizer

        if self.with_tv:
            ret['fg_tv'] = fg_tv

        return ret


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for _ in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]