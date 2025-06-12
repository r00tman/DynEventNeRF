import torch
from utils import TINY_NUMBER
from intersects import cylinder_intersect as cylinder_intersect_gold, cylinder_intersect


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise RuntimeError(
            f'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly! e.g. {p_norm_sq.max()}')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def intersect_cylinder_gold(ray_o, ray_d, r, h0, h1):
    '''
    ray_o, ray_d: [..., 3]
    compute the near and far depths of the intersection points between this ray and
    the cylinder at (0, 0, 0) with radius r and heights h0 and h1
    '''
    # todo: vectorize properly
    original_shape = ray_o.shape
    ray_o = ray_o.view(-1, 3)
    ray_d = ray_d.view(-1, 3)

    N = ray_o.shape[0]
    min_depths = ray_o.new_zeros(N)
    max_depths = ray_o.new_zeros(N)

    ray_o = ray_o.clone().cpu().numpy()
    ray_d = ray_d.clone().cpu().numpy()
    for i in range(N):
        o = ray_o[i]
        d = ray_d[i]
        res = cylinder_intersect_gold(o, d, r, h0, h1)
        if len(res) > 0:
            min_depths[i] = res[0]
            max_depths[i] = res[1]
        else:
            min_depths[i] = 0
            max_depths[i] = 2  # todo: check

    min_depths = min_depths.view(original_shape[:-1])
    max_depths = max_depths.view(original_shape[:-1])

    return min_depths, max_depths


def intersect_cylinder(ray_o, ray_d, r, h0, h1):
    '''
    ray_o, ray_d: [..., 3]
    compute the near and far depths of the intersection points between this ray and
    the cylinder at (0, 0, 0) with radius r and heights h0 and h1
    '''
    # todo: vectorize properly
    original_shape = ray_o.shape
    ray_o = ray_o.view(-1, 3)
    ray_d = ray_d.view(-1, 3)

    min_depths, max_depths = cylinder_intersect(ray_o, ray_d, r, h0, h1)

    min_depths = torch.nan_to_num(min_depths, 0, 0, 0)
    max_depths = torch.nan_to_num(max_depths, 2, 2, 2)

    min_depths = min_depths.view(original_shape[:-1])
    max_depths = max_depths.view(original_shape[:-1])

    return min_depths, max_depths


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]
    # todo: note that first and last samples are offset by 1/4 and -1/4 respectively on average,
    #       while all others are offset by 0 on averege

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)  # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)  # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00  # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1] * len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples, ])  # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf  # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)  # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]  # [..., N_samples]
    denom = torch.where(denom < TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples
