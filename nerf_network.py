from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn

from ddp_config import logger


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply


def get_activation_by_name(act, garf_sigma=1.0):
    if act == 'relu':
        # actclass = nn.ReLU
        actclass = MyReLU
    elif act == 'leaky_relu':
        # actclass = nn.ReLU
        actclass = MyLeakyReLU
    # elif act == 'sine':
    #     actclass = SineAct
    elif act == 'elu':
        actclass = nn.ELU
    elif act == 'tanh':
        actclass = nn.Tanh
    elif act == 'gelu':
        actclass = nn.GELU
    elif act == 'garf':
        actclass = lambda: MyGARF(garf_sigma)
    return actclass


class DummyEmbedder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = self.input_dim

    def forward(self, input_, iteration):
        '''
        :param input_: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert input_.shape[-1] == self.input_dim
        return input_

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos),
                       N_anneal=100000, N_anneal_min_freq=0,
                       use_annealing=True):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.use_annealing = use_annealing

        self.N_anneal = N_anneal
        self.N_anneal_min_freq = N_anneal_min_freq

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, iteration):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert input.shape[-1] == self.input_dim

        out = []
        if self.include_input:  # todo: wtf is with that?
            out.append(input)

        alpha = (len(self.freq_bands)-self.N_anneal_min_freq)*iteration/self.N_anneal
        for i in range(len(self.freq_bands)):
            w = (1-np.cos(np.pi*np.clip(alpha-i+self.N_anneal_min_freq, 0, 1)))/2.

            if not self.use_annealing:
                w = 1

            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq) * w)
        out = torch.cat(out, dim=-1)

        assert out.shape[-1] == self.out_dim
        return out


class TimeAwareEmbedder(nn.Module):
    def __init__(self, input_dim,
                       max_freq_log2_pos, max_freq_log2_time,
                       N_freqs_pos, N_freqs_time,
                       *args, **kwargs):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()
        assert input_dim == 4

        self.input_dim = input_dim

        self.embedder_pos = Embedder(3, max_freq_log2=max_freq_log2_pos, N_freqs=N_freqs_pos, *args, **kwargs)
        self.embedder_time = Embedder(1, max_freq_log2=max_freq_log2_time, N_freqs=N_freqs_time, *args, **kwargs)

        self.out_dim = self.embedder_pos.out_dim + self.embedder_time.out_dim

    def forward(self, input, iteration):
        assert input.shape[-1] == 4

        pos = input[..., :3]
        time = input[..., 3:]
        out_pos = self.embedder_pos(pos, iteration)
        out_time = self.embedder_time(time, iteration)
        out = torch.cat((out_pos, out_time), dim=-1)

        assert out.shape[-1] == self.out_dim
        return out




# default tensorflow initialization of linear layers
def weights_init(m, gain=5):
    if isinstance(m, nn.Linear):
        # GAIN = 5
        # GAIN = 8
        # GAIN = 50
        GAIN = gain
        nn.init.xavier_uniform_(m.weight.data, gain=GAIN)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, mean=0, std=GAIN)
        # if m.bias is not None:
        #     nn.init.zeros_(m.bias.data)


# class MyBatchNorm1d(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # self.bn = nn.BatchNorm1d(dim, momentum=0.01, affine=False)
#         self.bn = nn.LayerNorm(dim, elementwise_affine=False)

#     def forward(self, x):
#         return x

#         ordim = x.shape
#         # print(ordim)
#         x = x.view(-1, x.shape[-1])
#         x = self.bn(x)
#         return x.view(ordim)

class MyReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)*2

class MyLeakyReLU(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x)*2

# class MyTanh(nn.Module):
#     def forward(self, x):
#         return torch.tanh(x)*2

class MyGARF(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigmasq = sigma**2

    def forward(self, x):
        return torch.exp(-x**2/2/self.sigmasq)


class MLPNet(nn.Module):
    def __init__(self, D=8, W=256, input_ch=4, input_ch_viewdirs=3,
                 skips=[4], use_viewdirs=False, act='relu', garf_sigma=1.0,
                 crop_y=(-1.0, 1.0), crop_r=1.0, init_gain=5.0):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.skips = skips
        actclass = get_activation_by_name(act, garf_sigma=garf_sigma)

        self.crop_y = crop_y
        self.crop_r = crop_r

        self.base_layers = []
        dim = self.input_ch
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), actclass())
            )
            dim = W
            if i in self.skips and i != (D-1):      # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)
        my_init = lambda x: weights_init(x, gain=init_gain)
        # self.base_layers.apply(my_init)        # xavier init

        sigma_layers = [nn.Linear(dim, 1), ]       # sigma must be positive
        sigma_layers.append(nn.Softplus())
        # sigma_layers.append(nn.ReLU())
        self.sigma_layers = nn.Sequential(*sigma_layers)
        self.sigma_layers.apply(my_init)      # xavier init

        # rgb color
        rgb_layers = []
        base_remap_layers = [nn.Linear(dim, 256), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)
        # self.base_remap_layers.apply(my_init)

        dim = 256 + self.input_ch_viewdirs
        for i in range(1):
            rgb_layers.append(nn.Linear(dim, W // 2))
            rgb_layers.append(actclass())
            dim = W // 2
        rgb_layers.append(nn.Linear(dim, 3))
        rgb_layers.append(nn.Sigmoid())     # rgb values are normalized to [0, 1]
        # rgb_layers.append(nn.Softplus())     # rgb values are normalized to [0, inf]
        self.rgb_layers = nn.Sequential(*rgb_layers)
        # self.rgb_layers.apply(my_init)

    def forward(self, pts, viewdirs, iteration, embedder_position, embedder_viewdir):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        x, y, z, t = pts[..., 0], pts[..., 1], pts[..., 2], pts[..., 3]
        r2 =  x**2 + z**2
        mask = r2 <= self.crop_r**2
        mask = mask & (y >= self.crop_y[0]) & (y <= self.crop_y[1])

        # todo: treat time separately
        input_ = torch.cat((embedder_position(pts, iteration),
                           embedder_viewdir(viewdirs, iteration)), dim=-1)
        input_pts = input_[..., :self.input_ch]

        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers)-1):
            if i in self.skips:
                base = torch.cat((input_pts, base), dim=-1)
            base = self.base_layers[i+1](base)

        sigma = self.sigma_layers(base)
        sigma = torch.abs(sigma)

        # zero everything outside of the mask
        # todo: make it so it doesn't even compute nn for this
        sigma = sigma*mask[..., None]
        # print(mask.float().mean())

        base_remap = self.base_remap_layers(base)
        input_viewdirs = input_[..., -self.input_ch_viewdirs:]
        if not self.use_viewdirs:
            input_viewdirs = input_viewdirs * 0
        rgb = self.rgb_layers(torch.cat((base_remap, input_viewdirs), dim=-1))

        ret = OrderedDict([('rgb', rgb),
                           ('sigma', sigma.squeeze(-1))])
        return ret


class TCNNNet(nn.Module):
    def __init__(self, input_ch=4, input_ch_viewdirs=3, use_viewdirs=False,
                 crop_y=(-1.0, 1.0), crop_r=1.0, init_gain=5.0):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z, t)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs

        self.crop_y = crop_y
        self.crop_r = crop_r

        self.bound = 1
        per_level_scale = np.exp2(np.log2(2048 * self.bound / 16) / (16 - 1))
        self.encoder = tcnn.Encoding(
            n_input_dims=self.input_ch,
            encoding_config={
                "otype": "HashGrid",
                # "n_levels": 16,
                "n_levels": 8,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                # "base_resolution": 16,
                "base_resolution": 8,
                # "per_level_scale": per_level_scale,
                "per_level_scale": 2.0,
            }
        )

        # self.geo_feat_dims = 15
        self.geo_feat_dims = 10
        # self.hidden_dim = 64
        self.hidden_dim = 16
        self.num_layers = 2

        self.sigma_net = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims,
            n_output_dims=1 + self.geo_feat_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers -1,
            }
        )

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.hidden_dim_color = 64
        self.num_layers_color = 3
        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dims

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_color,
                "n_hidden_layers": self.num_layers_color - 1,
            },
        )


    def forward(self, pts, viewdirs, iteration, embedder_position, embedder_viewdir):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        batch_shape = pts.shape[:-1]
        pts = pts.reshape(-1, pts.shape[-1])
        viewdirs = viewdirs.reshape(-1, viewdirs.shape[-1])

        x, y, z, t = pts[..., 0], pts[..., 1], pts[..., 2], pts[..., 3]
        r2 =  x**2 + z**2
        crop_r = self.crop_r * min(iteration, 10000) / 10000
        mask = r2 <= crop_r**2
        mask = mask & (y >= self.crop_y[0]) & (y <= self.crop_y[1])

        # todo: treat time properly
        pts[:3] = (pts[:3] + self.bound) / (2 * self.bound)  # from [-1, 1] to [0, 1]
        pts = self.encoder(pts)

        h = self.sigma_net(pts)
        sigma = trunc_exp(h[..., 0])
        sigma = sigma * mask

        geo_feat = h[..., 1:]

        if not self.use_viewdirs:
            viewdirs = viewdirs * 0

        d = (viewdirs + 1) / 2
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        color = torch.sigmoid(h)

        color = color.reshape(batch_shape+(3,))
        sigma = sigma.reshape(batch_shape+(1,))

        ret = OrderedDict([('rgb', color),
                           ('sigma', sigma.squeeze(-1))])
        return ret


def sample_vec(vec, coord):
    assert len(coord.shape) == 1

    N = coord.shape[0]
    coord = torch.stack((torch.zeros_like(coord), coord), -1)
    coord = coord.view(1, -1, 1, 2)

    # res = F.grid_sample(vec, coord, align_corners=True)
    res = F.grid_sample(vec, coord, align_corners=False)
    # transform (1, F, N, 1) into (N, F, 1, 1), then into (N, F)
    res = res.permute(2, 1, 0, 3).squeeze(3).squeeze(2)

    assert len(res.shape) == 2
    assert res.shape[0] == N

    return res

def resize_bilinear(input_matrix, new_size):
    # Resize the input matrix using bilinear interpolation
    # input NxM -> output new_size
    # resized_matrix = F.interpolate(input_matrix.unsqueeze(0).unsqueeze(0), size=new_size, mode='bilinear', align_corners=True)
    resized_matrix = F.interpolate(input_matrix.unsqueeze(0).unsqueeze(0), size=new_size, mode='bilinear', align_corners=False)
    return resized_matrix.squeeze()


def resize_vec(newvec, oldvec):
    # 1 x FR x N x 1
    newN = newvec.shape[2]
    oldN = oldvec.shape[2]
    oldFR = oldvec.shape[1]

    oldvec = oldvec.squeeze(3).squeeze(0)
    new_size = (oldFR, newN)
    resized = resize_bilinear(oldvec, new_size)
    resized = resized.unsqueeze(0).unsqueeze(3)

    assert resized.shape == (1, oldFR, newN, 1)

    newvec.copy_(resized)



def sample_mat(mat, coord):
    assert len(coord.shape) == 2
    assert coord.shape[1] == 2

    N = coord.shape[0]
    coord = coord.view(1, -1, 1, 2)

    res = F.grid_sample(mat, coord, align_corners=True)
    # transform (1, F, N, 1) into (N, F, 1, 1), then into (N, F)
    res = res.permute(2, 1, 0, 3).squeeze(3).squeeze(2)

    assert len(res.shape) == 2
    assert res.shape[0] == N

    return res


class TensoRFVMNet(nn.Module):
    def __init__(self, N=500, Ntime=24, F=1+3, R=1,
                 input_ch=3, input_ch_viewdirs=3, use_viewdirs=False,
                 crop_y=(-1.0, 1.0), crop_r=1.0, init_gain=1.0):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        assert input_ch == 3
        assert input_ch_viewdirs == 3
        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs

        self.crop_y = crop_y
        self.crop_r = crop_r

        self.bound = 1

        self.N = N  # density of the grid
        self.Ntime = Ntime  # density of the grid across time
        self.F = F  # number of features
        self.R = R  # rank

        self.xvec = nn.Parameter(torch.randn(1, F*R, N, 1)/R*init_gain)
        self.yvec = nn.Parameter(torch.randn(1, F*R, N, 1)/R*init_gain)
        self.zvec = nn.Parameter(torch.randn(1, F*R, N, 1)/R*init_gain)

        self.YZmat = nn.Parameter(torch.randn(1, F*R, N, N)/R*init_gain)
        self.XZmat = nn.Parameter(torch.randn(1, F*R, N, N)/R*init_gain)
        self.XYmat = nn.Parameter(torch.randn(1, F*R, N, N)/R*init_gain)

        self.XTmat = nn.Parameter(torch.randn(1, F*R, N, Ntime)/R*init_gain)
        self.YTmat = nn.Parameter(torch.randn(1, F*R, N, Ntime)/R*init_gain)
        self.ZTmat = nn.Parameter(torch.randn(1, F*R, N, Ntime)/R*init_gain)



    def forward(self, pts, viewdirs, iteration, embedder_position, embedder_viewdir):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        batch_shape = pts.shape[:-1]
        pts = pts.reshape(-1, pts.shape[-1])
        viewdirs = viewdirs.reshape(-1, viewdirs.shape[-1])

        x, y, z, t = pts[..., 0], pts[..., 1], pts[..., 2], pts[..., 3]
        r2 = x**2 + z**2
        # crop_r = self.crop_r * min(iteration, 20000) / 20000
        crop_r = self.crop_r
        mask = r2 <= crop_r**2
        mask = mask & (y >= self.crop_y[0]) & (y <= self.crop_y[1])

        # normalize the coordinates according to the crop region
        # todo: do it isotropically
        # bound = max(self.crop_r, abs(self.crop_y[0]), abs(self.crop_y[1]))
        # pts = pts / bound
        pts = pts.clone()
        pts[..., (0, 2)] = pts[..., (0, 2)] / self.crop_r # already [-1, 1]
        pts[..., 1] = (pts[..., 1] - self.crop_y[0]) / (self.crop_y[1] - self.crop_y[0])
        pts[..., 1] = pts[..., 1] * 2 - 1 # normalize from [0, 1] to [-1, 1]
        pts[..., 3] = pts[..., 3] * 2 - 1 # time from [0, 1] to [-1, 1]

        # pts = (pts + self.bound) / (2 * self.bound)  # from [-1, 1] to [0, 1]

        # this expects (1, hout, wout, 2) and returns (1, c, hout, wout)
        # for vector, we set wout = 1, hout = -1, and we get (1, c, -1, 1)

        xv = sample_vec(self.xvec, pts[..., 0])
        yv = sample_vec(self.yvec, pts[..., 1])
        zv = sample_vec(self.zvec, pts[..., 2])

        YZm = sample_mat(self.YZmat, pts[..., (1, 2)])
        XZm = sample_mat(self.XZmat, pts[..., (0, 2)])
        XYm = sample_mat(self.XYmat, pts[..., (0, 1)])

        XTm = sample_mat(self.XTmat, pts[..., (0, 3)])
        YTm = sample_mat(self.YTmat, pts[..., (1, 3)])
        ZTm = sample_mat(self.ZTmat, pts[..., (2, 3)])

        # res = xv * YZm + yv * XZm + zv * XYm
        # source: HexPlane formulation https://caoang327.github.io/HexPlane/HexPlane.pdf
        res = XYm * ZTm * xv + XZm * YTm * yv + YZm * XTm * zv
        res = res.view(pts.shape[0], self.F, self.R).sum(-1)
        assert res.shape == (pts.shape[0], self.F)

        sigma = res[..., 0]
        sigma = trunc_exp(sigma)

        sigma = sigma * mask

        # if self.use_viewdirs == False:
        #     viewdirs = viewdirs * 0

        # d = (viewdirs + 1) / 2
        # d = self.encoder_dir(d)

        # h = torch.cat([d, geo_feat], dim=-1)
        # h = self.color_net(h)

        color = res[..., 1:]

        color = torch.sigmoid(color)

        color = color.reshape(batch_shape+(3,))
        sigma = sigma.reshape(batch_shape+(1,))

        ret = OrderedDict([('rgb', color),
                           ('sigma', sigma.squeeze(-1))])
        return ret

    def get_sparsity_reg(self):
        res = 0

        res = res + abs(self.xvec).mean()
        res = res + abs(self.yvec).mean()
        res = res + abs(self.zvec).mean()

        res = res + abs(self.YZmat).mean()
        res = res + abs(self.XZmat).mean()
        res = res + abs(self.XYmat).mean()

        res = res + abs(self.XTmat).mean()
        res = res + abs(self.YTmat).mean()
        res = res + abs(self.ZTmat).mean()

        return res


class TensoRFCPNet(nn.Module):
    def __init__(self, N=500, Ntime=24, F=1+3, R=100,
                 Hsteps=10, Hmin=16, Hmin_time=16, Hiters=2000,
                 input_ch=3, input_ch_viewdirs=3, use_viewdirs=False,
                 crop_y=(-1.0, 1.0), crop_r=1.0, init_gain=1.0):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        assert input_ch == 3
        assert input_ch_viewdirs == 3
        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs

        self.crop_y = crop_y
        self.crop_r = crop_r

        self.bound = 1

        self.N = N  # density of the grid
        self.Ntime = Ntime  # density of the grid across time
        self.F = F  # number of features
        self.R = R  # rank

        self.xvec_hh = nn.ParameterList()
        self.yvec_hh = nn.ParameterList()
        self.zvec_hh = nn.ParameterList()
        self.tvec_hh = nn.ParameterList()

        self.Hsteps = Hsteps
        self.Hmin = Hmin
        self.Hmin_time = Hmin_time
        self.Hiters = Hiters
        for i, Nc in enumerate(np.geomspace(self.Hmin, N, num=self.Hsteps)):
            Nc = int(Nc)
            self.xvec_hh.append(nn.Parameter(torch.randn(1, F*R, Nc, 1)/R*init_gain))
            self.yvec_hh.append(nn.Parameter(torch.randn(1, F*R, Nc, 1)/R*init_gain))
            self.zvec_hh.append(nn.Parameter(torch.randn(1, F*R, Nc, 1)/R*init_gain))

        for i, Nc in enumerate(np.geomspace(self.Hmin_time, Ntime, num=self.Hsteps)):
            Nc = int(Nc)
            self.tvec_hh.append(nn.Parameter(torch.randn(1, F*R, Nc, 1)/R*init_gain))

        # self.xvec = nn.Parameter(torch.randn(1, F*R, N, 1)/R*init_gain)
        # self.yvec = nn.Parameter(torch.randn(1, F*R, N, 1)/R*init_gain)
        # self.zvec = nn.Parameter(torch.randn(1, F*R, N, 1)/R*init_gain)
        self.register_buffer('active_idx', torch.tensor(0))


    def rescale_grid(self, new_idx):
        old_idx = self.active_idx.item()
        logger.warning(f'resizing from {old_idx} ({self.xvec_hh[old_idx].shape}) to {new_idx} ({self.xvec_hh[new_idx].shape})')

        resize_vec(self.xvec_hh[new_idx].data, self.xvec_hh[old_idx].data)
        resize_vec(self.yvec_hh[new_idx].data, self.yvec_hh[old_idx].data)
        resize_vec(self.zvec_hh[new_idx].data, self.zvec_hh[old_idx].data)
        resize_vec(self.tvec_hh[new_idx].data, self.tvec_hh[old_idx].data)

        self.active_idx.fill_(new_idx)


    def forward(self, pts, viewdirs, iteration, embedder_position, embedder_viewdir):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        batch_shape = pts.shape[:-1]
        pts = pts.reshape(-1, pts.shape[-1])
        viewdirs = viewdirs.reshape(-1, viewdirs.shape[-1])

        x, y, z, t = pts[..., 0], pts[..., 1], pts[..., 2], pts[..., 3]
        r2 = x**2 + z**2
        # crop_r = self.crop_r * min(iteration, 20000) / 20000
        crop_r = self.crop_r
        mask = r2 <= crop_r**2
        mask = mask & (y >= self.crop_y[0]) & (y <= self.crop_y[1])

        # normalize the coordinates according to the crop region
        # todo: do it isotropically
        # bound = max(self.crop_r, abs(self.crop_y[0]), abs(self.crop_y[1]))
        # pts = pts / bound
        pts = pts.clone()
        pts[..., (0, 2)] = pts[..., (0, 2)] / self.crop_r  # from -1 to 1
        pts[..., 1] = (pts[..., 1] - self.crop_y[0]) / (self.crop_y[1] - self.crop_y[0])  # from 0 to 1
        pts[..., 1] = pts[..., 1] * 2 - 1 # from -1 to 1
        pts[..., 3] = pts[..., 3] * 2 - 1 # time: from -1 to 1

        # pts = (pts + self.bound) / (2 * self.bound)  # from [-1, 1] to [0, 1]

        # this expects (1, hout, wout, 2) and returns (1, c, hout, wout)
        # for vector, we set wout = 1, hout = -1, and we get (1, c, -1, 1)

        supposed_idx = min(self.Hsteps-1, iteration*self.Hsteps//self.Hiters)
        if supposed_idx != self.active_idx.item():
            self.rescale_grid(supposed_idx)

        xvec = self.xvec_hh[supposed_idx]
        yvec = self.yvec_hh[supposed_idx]
        zvec = self.zvec_hh[supposed_idx]
        tvec = self.tvec_hh[supposed_idx]

        xv = sample_vec(xvec, pts[..., 0])
        yv = sample_vec(yvec, pts[..., 1])
        zv = sample_vec(zvec, pts[..., 2])
        tv = sample_vec(tvec, pts[..., 3])

        res = (xv * yv * zv * tv).view(pts.shape[0], self.F, self.R).sum(-1)
        assert res.shape == (pts.shape[0], self.F)

        sigma = res[..., 0]
        sigma = trunc_exp(sigma)

        sigma = sigma * mask

        # if self.use_viewdirs == False:
        #     viewdirs = viewdirs * 0

        # d = (viewdirs + 1) / 2
        # d = self.encoder_dir(d)

        # h = torch.cat([d, geo_feat], dim=-1)
        # h = self.color_net(h)

        color = res[..., 1:]

        color = torch.sigmoid(color)

        color = color.reshape(batch_shape+(3,))
        sigma = sigma.reshape(batch_shape+(1,))

        ret = OrderedDict([('rgb', color),
                           ('sigma', sigma.squeeze(-1))])
        return ret

    def get_sparsity_reg(self):
        res = 0

        active_idx = self.active_idx.item()
        xvec = self.xvec_hh[active_idx]
        yvec = self.yvec_hh[active_idx]
        zvec = self.zvec_hh[active_idx]
        tvec = self.tvec_hh[active_idx]

        res = res + abs(xvec).mean()
        res = res + abs(yvec).mean()
        res = res + abs(zvec).mean()
        res = res + abs(tvec).mean()

        return res

    def get_smoothness_reg(self):
        res = 0

        active_idx = self.active_idx.item()
        xvec = self.xvec_hh[active_idx]
        yvec = self.yvec_hh[active_idx]
        zvec = self.zvec_hh[active_idx]
        tvec = self.tvec_hh[active_idx]

        # [1, F*R, N, 1] ---> [N, F*R]
        xvec = xvec.squeeze(3).squeeze(0).T
        yvec = yvec.squeeze(3).squeeze(0).T
        zvec = zvec.squeeze(3).squeeze(0).T
        tvec = tvec.squeeze(3).squeeze(0).T
        # todo: check that it is correct

        res = res + (torch.diff(xvec, dim=0)**2).mean()
        res = res + (torch.diff(yvec, dim=0)**2).mean()
        res = res + (torch.diff(zvec, dim=0)**2).mean()
        res = res + (torch.diff(tvec, dim=0)**2).mean()

        return res

    def get_tv_reg(self):
        res = 0

        active_idx = self.active_idx.item()
        xvec = self.xvec_hh[active_idx]
        yvec = self.yvec_hh[active_idx]
        zvec = self.zvec_hh[active_idx]
        tvec = self.tvec_hh[active_idx]

        # [1, F*R, N, 1] ---> [N, F*R]
        xvec = xvec.squeeze(3).squeeze(0).T
        yvec = yvec.squeeze(3).squeeze(0).T
        zvec = zvec.squeeze(3).squeeze(0).T
        tvec = tvec.squeeze(3).squeeze(0).T
        # todo: check that it is correct

        res = res + abs(torch.diff(xvec, dim=0)).mean()
        res = res + abs(torch.diff(yvec, dim=0)).mean()
        res = res + abs(torch.diff(zvec, dim=0)).mean()
        res = res + abs(torch.diff(tvec, dim=0)).mean()

        return res
