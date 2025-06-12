from collections import OrderedDict
import torch
from torch.cuda.amp import autocast
from ddp_sampling import intersect_cylinder, intersect_sphere, sample_pdf


def render_single_image(rank, world_size, models, ray_sampler, chunk_size, iteration, args, timestamp=0.0):
    ##### parallel rendering of a single image
    ray_sampler.update_rays(models['camera_mgr'])
    ray_batch = ray_sampler.get_all(timestamp)

    fixed = 0
    if (ray_batch['ray_d'].shape[0] // world_size) * world_size != ray_batch['ray_d'].shape[0]:
        fixed = world_size - (ray_batch['ray_d'].shape[0] % world_size)
        for p in ray_batch:
            if ray_batch[p] is not None:
                ray_batch[p] = torch.cat((ray_batch[p], ray_batch[p][-fixed:]), dim=0)
    #     raise Exception('Number of pixels in the image is not divisible by the number of GPUs!\n\t# pixels: {}\n\t# GPUs: {}'.format(ray_batch['ray_d'].shape[0],
    #                                                                                                                                  world_size))

    # split into ranks; make sure different processes don't overlap
    # world_size = 1, therefore split_size is always the full shape
    # rank_split_sizes = [ray_batch['ray_d'].shape[0] // world_size, ] * world_size
    # rank_split_sizes[-1] = ray_batch['ray_d'].shape[0] - sum(rank_split_sizes[:-1])
    rank_split_sizes = [ray_batch['ray_d'].shape[0]]
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch[key] = torch.split(ray_batch[key], rank_split_sizes)[rank].to(rank)

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        ray_t = ray_batch_split['ray_t'][s]
        min_depth = ray_batch_split['min_depth'][s]
        background_linear = ray_batch_split['background_linear'][s] if 'background_linear' in ray_batch_split else None

        net = models['net']
        for m in range(models['cascade_level']):
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]

                # ----
                fg_near_depth_cyl, fg_far_depth_cyl = intersect_cylinder(ray_o, ray_d, args.crop_r, args.crop_y_min, args.crop_y_max)
                assert fg_far_depth_cyl.shape == fg_far_depth_cyl.shape
                assert fg_near_depth_cyl.shape == fg_near_depth_cyl.shape

                # print(fg_near_depth, fg_near_depth_cyl)
                fg_near_depth = torch.maximum(fg_near_depth, fg_near_depth_cyl)
                # print(fg_far_depth, fg_far_depth_cyl)
                fg_far_depth = torch.minimum(fg_far_depth, fg_far_depth_cyl)
                # ----

                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                # background depth
                # bg_depth = torch.linspace(0., 1., N_samples).view(
                #     [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).to(rank)

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                # bg_weights = ret['bg_weights'].clone().detach()
                # bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                # bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                # bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                #                               N_samples=N_samples, det=True)  # [..., N_samples]
                # bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples
                # del bg_weights
                # del bg_depth_mid
                # del bg_depth_samples
                torch.cuda.empty_cache()

            with autocast():
                with torch.no_grad():
                    ret = net(ray_o, ray_d, ray_t, fg_far_depth, fg_depth, background_linear, iteration)

            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()

    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0)

    # merge results from different processes
    if rank == 0:
        ret_merge_rank = [OrderedDict() for _ in range(len(ret_merge_chunk))]
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                # generate tensors to store results from other processes
                # ret_merge_rank[m][key] = [torch.zeros(*[size, ] + sh, dtype=torch.float32) for size in rank_split_sizes]
                # torch.distributed.gather(ret_merge_chunk[m][key], ret_merge_rank[m][key])
                # ret_merge_rank[m][key] = torch.cat(ret_merge_rank[m][key], dim=0)
                ret_merge_rank[m][key] = ret_merge_chunk[m][key]
                if fixed > 0:
                    ret_merge_rank[m][key] = ret_merge_rank[m][key][:-fixed]
                ret_merge_rank[m][key] = ret_merge_rank[m][key].reshape(
                    (ray_sampler.H, ray_sampler.W, -1)).squeeze()
                # print(m, key, ret_merge_rank[m][key].shape)
    else:  # send results to main process
        pass
        # for m in range(len(ret_merge_chunk)):
        #     for key in ret_merge_chunk[m]:
        #         torch.distributed.gather(ret_merge_chunk[m][key])

    # only rank 0 program returns
    if rank == 0:
        return ret_merge_rank
    else:
        return None
