import os
import os.path
from collections import OrderedDict
import torch
import torch.optim
from ddp_model import NerfNet
from ddp_config import logger


def create_nerf(rank, args, camera_mgr, load_camera_mgr=True, load_optimizer=True, use_lr_scheduler=True):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777+args.seed_offset)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]

    net = NerfNet(args).to(rank)
    # net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    # net = DDP(net, device_ids=[rank], output_device=rank)
    # optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    if use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda it: it/4000 if it < 4000 else 0.95**(it/10000))
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda it: 1.0)
    models['net'] = net
    models['optim'] = optim
    models['lr_scheduler'] = lr_scheduler

    models['camera_mgr'] = camera_mgr.to(rank)

    start = -1

    ###### load pretrained weights; each process should do this
    # if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
    #     ckpts = [args.ckpt_path]
    # else:
    if args.force_ckpt_path and args.force_ckpt_path != 'None':  # todo: figure out why it is sometimes 'None' in the string form
        logger.warning(f'Forcing this checkpoint: {args.force_ckpt_path}')
        ckpts = [args.force_ckpt_path]
    else:
        if args.init_ckpt_path is not None and len(args.init_ckpt_path) > 0:
            ckpts = [args.init_ckpt_path]
        else:
            ckpts = []


        new_ckpts = [os.path.join(args.basedir, args.expname, f)
                     for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]

        if len(new_ckpts) > 0:
            # newer checkpoints of this particular experiment already exist and should be used
            ckpts = new_ckpts
        else:
            # this is the first time we train the model, we should use init_ckpt_path
            load_camera_mgr = False
            load_optimizer = False
            logger.info(f'Initializing the network using: {args.init_ckpt_path}')
            logger.info('NOT loading optimizer and camera manager parameters')


    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])

    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        logger.info(f'load_camera_mgr={load_camera_mgr} load_optimizer={load_optimizer}')
        if (args.force_ckpt_path and args.force_ckpt_path != 'None') or len(new_ckpts) > 0:
            # only if it's reloading newer checkpoints of this particular experiment
            # otherwise, the iteration value is useless
            start = path2iter(fpath)

        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)

        names = ['net']
        if load_optimizer:
            names.append('optim')
            names.append('lr_scheduler')

        for name in names:
            if name == 'net':
                # print(name) # bound_transform.translation
                if 'bound_transform.translation' not in to_load[name]:
                    to_load[name]['bound_transform.translation'] = models[name].bound_transform.translation
            if name == 'lr_scheduler' and name not in to_load:
                models[name].last_epoch = start
                continue
            models[name].load_state_dict(to_load[name])  # todo: remove strict

        if load_camera_mgr:
            name = 'camera_mgr'
            models[name].load_state_dict(to_load[name])

    logger.info(f'start={start}')
    return start, models
