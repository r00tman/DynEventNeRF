#!/usr/bin/env python3
import os
from os import path
import glob
import sys
import re
from typing import List

import configargparse
import numpy as np
from PIL import Image
from tqdm import tqdm


parser = configargparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, help='data directory',
        default='data/dynsyn/lego_dyn2/groundtruth/train/')

parser.add_argument('-o', '--output', type=str, help='output directory',
        default='data/dynsyn/lego_dyn2/groundtruth/train_blurry32fps/')

parser.add_argument('-f', '--frame_list', type=str, help='frame list to render (renders all found frames if not specified)',
        default='frame_lists/33_frames.txt')


args = parser.parse_args()


def find_frames(img_path, view):
    result = glob.glob(path.join(img_path, f'r_{view}_????.png'))
    result.sort()
    return result

def load_image(fn: str):
    a = np.array(Image.open(fn))
    a = a / 255.
    return a

def make_blurry(linear_frames: List[np.ndarray]):
    return np.mean(linear_frames, 0)

def srgb_to_linear(x: np.ndarray):
    x = np.asarray(x)
    linmask = x <= 0.04045
    linval = x/12.92
    expval = ((x+0.055)/1.055)**2.4
    res = linval*linmask+(~linmask)*expval
    return res

def linear_to_srgb(x: np.ndarray):
    x = np.asarray(x)
    linmask = x <= 0.0031308
    linval = x*12.92
    expval = np.maximum(x, 0)**(1/2.4)*1.055-0.055
    res = linval*linmask+(~linmask)*expval
    return res

def save_result(result: np.ndarray, out_path: str, view: str, frame_id: int):
    result = np.clip(result * 255., 0, 255).astype(np.uint8)

    fn = path.join(out_path, f'r_{view}_{frame_id:04d}.png')
    os.makedirs(out_path, exist_ok=True)
    Image.fromarray(result).save(fn)



# # srgb transform self-test
# for v in np.linspace(0, 1, num=200):
#     a = v
#     print(a)
#     a = srgb_to_linear(a)
#     print(a)
#     a = linear_to_srgb(a)
#     print(a, a-v)
#     print()
# 1/0

# top-down:
# for each view independently?:
#   find and load all frames
#   find frames from the frame_list and split into groups
#   take groups and make blurry
#   save them in the results directory

# breakpoint()

with open(args.frame_list) as f:
    frame_list = [int(l.strip()) for l in f.readlines() if l.strip()]
print('found', len(frame_list), 'frames in the list')

for view in ['00', '01', '02', '03', '04']:
    print('processing view', view)
    frame_fns = find_frames(args.input, view)
    print('loading', len(frame_fns), 'images...')
    frames = [load_image(frame) for frame in tqdm(frame_fns)]
    print('linearizing frames...')
    linear_frames = [srgb_to_linear(frame) for frame in tqdm(frames)]
    group = []
    i = 0
    frame_list_idx = 0
    while i < len(frames):
        group.append(linear_frames[i])
        if frame_list[frame_list_idx] == i:
            linear_result = make_blurry(group)
            result = linear_to_srgb(linear_result)
            print('saving blurred frame', i)
            save_result(result, args.output, view, i)
            group = []
            frame_list_idx += 1
        i = i + 1
