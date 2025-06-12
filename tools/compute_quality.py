#!/usr/bin/env python3
from os import path
from glob import glob
import re

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm
import torch
import configargparse
import lpips

lpips_loss = lpips.LPIPS(net='alex')


parser = configargparse.ArgumentParser()

parser.add_argument('-w', '--work_dir', type=str, help='predictions directory',
        default='logs_auto/auto_slurm_6kits')

parser.add_argument('-g', '--gt_dir', type=str, help='ground truth directory',
        default='data/dynsyn/lego_dyn2/groundtruth')

parser.add_argument('-s', '--splits', type=str, help='splits',
        default='train,validation')

parser.add_argument('-f', '--frame_list', type=str, help='frame list to render (renders all found frames if not specified)',
        default=None)

# parser.add_argument('-r', '--relaxed', action='store_true', help='don\'t fail if not all frames of frame list are found',
#         default=None)

args = parser.parse_args()

# work_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs_auto/auto_slurm'
# work_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs_auto/auto_slurm_notrfloss'
# work_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs_auto/auto_slurm_6kits'
work_dir = args.work_dir
# gt_dir = 'data/dynsyn/lego_dyn2/groundtruth'
gt_dir = args.gt_dir

# frame_list = sys.argv[2] if len(sys.argv) > 2 else None
frame_list = args.frame_list
# frame_list = 'frame_lists/1_frames.txt'
# frame_list = 'frame_lists/17_frames.txt'
if frame_list:
    print('using frame list:', frame_list)
    with open(frame_list) as f:
        frame_list = [int(l.strip()) for l in f.readlines() if l.strip()]
    print('found', len(frame_list), 'frames', 'in the list')

globs_by_split = {
    'test': [
        ('test00050', 'train_*/render_validation_*/r_00050.png'),
        ('test00383', 'train_*/render_validation_*/r_00383.png'),
        ('test00716', 'train_*/render_validation_*/r_00716.png'),
    ],

    'train': [
        ('train00', 'train_*/render_train_*/r_00_0000.png'),
        ('train01', 'train_*/render_train_*/r_01_0000.png'),
        ('train02', 'train_*/render_train_*/r_02_0000.png'),
        ('train03', 'train_*/render_train_*/r_03_0000.png'),
        ('train04', 'train_*/render_train_*/r_04_0000.png'),
    ]
}

gt_globs_by_split = {
    'test': [
        ('test00050', 'validation/r_0050_????.png'),
        ('test00383', 'validation/r_0383_????.png'),
        ('test00716', 'validation/r_0716_????.png'),
    ],

    'train': [
        ('train00', 'train/r_00_????.png'),
        ('train01', 'train/r_01_????.png'),
        ('train02', 'train/r_02_????.png'),
        ('train03', 'train/r_03_????.png'),
        ('train04', 'train/r_04_????.png'),
    ]
}

globs = []
gt_globs = []
for split in args.splits.split(','):
    globs += globs_by_split[split]
    gt_globs += gt_globs_by_split[split]

def extract_frame_id_from_pred(x):
    # s = re.search(r'frame_([0-9]*)', x)
    s = re.search(r'train_([0-9]*)', x)
    if s is None:
        return None
    else:
        return int(s.group(1))

def extract_frame_id_from_gt(x):
    # s = re.search(r'frame_([0-9]*)', x)
    s = re.search(r'_([0-9]*).png', x)
    if s is None:
        return None
    else:
        return int(s.group(1))

def compute_psnr(pred, gt):
    # todo: try prequantizing gt files or optimizing the Ax+b color transform?

    psnr = -10*np.log10(np.mean((pred-gt)**2))
    return psnr

def compute_ssim(pred, gt):
    ssim = structural_similarity(pred, gt, multichannel=True, channel_axis=2, data_range=1.)
    return ssim

def preprocess_lpips(a):
    # normalize to [-1, 1]
    a = a * 2 - 1
    # swap channels
    a = np.transpose(a, (2, 0, 1))[None, ...]
    a = torch.from_numpy(a).float()
    return a

def compute_lpips(pred, gt):
    pred = preprocess_lpips(pred)
    gt = preprocess_lpips(gt)

    res = lpips_loss(pred, gt).item()
    return res

group_psnr = dict()
group_ssim = dict()
group_lpips = dict()

for (group_name, suffix), (group_name_, gt_suffix) in zip(globs, gt_globs):
    assert group_name == group_name_

    # find pngs
    preds = glob(path.join(work_dir, suffix))
    gts = glob(path.join(gt_dir, gt_suffix))
    preds.sort()
    gts.sort()

    # extract frame numbers
    preds = [(extract_frame_id_from_pred(fn), fn) for fn in preds]
    gts = [(extract_frame_id_from_gt(fn), fn) for fn in gts]

    # convert to dict
    preds_dict = dict(preds)
    gts_dict = dict(gts)
    assert len(preds_dict) == len(preds), 'there should be no duplicates with the same frame numbers among preds'
    assert len(gts_dict) == len(gts), 'there should be no duplicates with the same frame numbers among gts'

    preds_numbers = set(preds_dict.keys())
    gts_numbers = set(gts_dict.keys())

    common_numbers = preds_numbers.intersection(gts_numbers)
    print(group_name, '#common frames:', len(common_numbers), '#predicted frames:', len(preds_numbers), '#gt frames:', len(gts_numbers))

    if frame_list:
        common_numbers = common_numbers.intersection(frame_list)
        print('applied frame list of', len(frame_list), 'frames')
        assert len(common_numbers) == len(frame_list), f'all frame list frames must exist: intersection is {len(common_numbers)} frames, frame list is {len(frame_list)} frames'

    psnr_dict = dict()
    ssim_dict = dict()
    lpips_dict = dict()

    for number in tqdm(list(sorted(common_numbers))):
        pred_fn = preds_dict[number]
        gt_fn = gts_dict[number]

        pred = np.array(Image.open(pred_fn))[:, :, :3]/255.
        gt = np.array(Image.open(gt_fn))[:, :, :3]/255.
        assert pred.shape == gt.shape, f'shapes must match: pred has {pred.shape}, gt has {gt.shape}'

        psnr = compute_psnr(pred, gt)
        psnr_dict[number] = psnr
        # print(group_name, 'frame:', number, 'psnr:', psnr)

        ssim = compute_ssim(pred, gt)
        ssim_dict[number] = ssim
        # print(group_name, 'frame:', number, 'ssim:', ssim)

        lpips = compute_lpips(pred, gt)
        lpips_dict[number] = lpips
        # print(group_name, 'frame:', number, 'lpips:', lpips)

    average_psnr = np.mean(list(psnr_dict.values()))
    average_ssim = np.mean(list(ssim_dict.values()))
    average_lpips = np.mean(list(lpips_dict.values()))

    group_psnr[group_name] = average_psnr
    group_ssim[group_name] = average_ssim
    group_lpips[group_name] = average_lpips

    # print(group_name, 'average psnr:', average_psnr)
    # print(group_name, 'average ssim:', average_ssim)
    # print(group_name, 'average lpips:', average_lpips)

for split in args.splits.split(','):
    print()
    print(f'average {split} psnr:', np.mean([val for grp, val in group_psnr.items() if split in grp]))
    print(f'average {split} ssim:', np.mean([val for grp, val in group_ssim.items() if split in grp]))
    print(f'average {split} lpips:', np.mean([val for grp, val in group_lpips.items() if split in grp]))
