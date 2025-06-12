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
import ffmpeg
from tqdm import tqdm


parser = configargparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, help='data directory',
        default='data/dynsyn/lego_dyn2/groundtruth/train_blurry32fps/')

parser.add_argument('-o', '--output', type=str, help='output fn',
        default='data/dynsyn/lego_dyn2/groundtruth/train_blurry32fps.mp4')

parser.add_argument('-f', '--frame_list', type=str, help='frame list to render (renders all found frames if not specified)',
        default=None)


args = parser.parse_args()

def find_frames(img_path, view):
    result = glob.glob(path.join(img_path, f'r_{view}_????.png'))
    result.sort()
    return result

def load_image(fn: str):
    a = np.array(Image.open(fn))[..., :3]  # RGB
    return a


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
frame_list = args.frame_list
if frame_list:
    with open(frame_list) as f:
        frame_list = [int(l.strip()) for l in f.readlines() if l.strip()]
    print('found', len(frame_list), 'frames in the list')

views = ['00', '01', '02', '03', '04']
frames_by_views_dict = {view: find_frames(args.input, view) for view in views}
N_frames = len(frames_by_views_dict[views[0]])
assert all(len(v) == N_frames for v in frames_by_views_dict.values())

frames_by_views = [[frames_by_views_dict[view][idx] for view in views] for idx in range(N_frames)]

def compose_frame(views):
    IMH, IMW, IMC = views[0].shape

    H = 2
    W = 3
    pad = views[0]*0
    # while len(views) < H*W:
    #     views.append(pad)

    rows = []
    # breakpoint()
    for rowidx in range(H):
        rowsrc = views[rowidx*W:(rowidx+1)*W]
        if len(rowsrc) < W:
            pad = np.zeros((IMH, (W-len(rowsrc))*IMW//2, 3), dtype=np.uint8)
            rowsrc = [pad] + rowsrc + [pad]
        row = np.concatenate(rowsrc, 1)
        row = row.reshape(IMH, IMW*W, IMC)
        rows.append(row)
    result = np.concatenate(rows, 0)

    return result

process2 = None

frame_list_idx = 0

for idx, frame in enumerate(frames_by_views):
    if frame_list and frame_list_idx < len(frame_list) and frame_list[frame_list_idx] != idx:
        continue
    frame_list_idx += 1

    print('processing frame', idx, frame)
    frames = [load_image(frame) for frame in frame]
    cframe = compose_frame(frames)

    if process2 is None:
        assert len(cframe.shape) == 3  # H, W, C
        assert cframe.shape[2] == 3  # RGB

        # Image.fromarray(cframe).save('test.png')
        width, height = cframe.shape[1], cframe.shape[0]
        process2 = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=30)
            .output(path.join(args.output), pix_fmt='yuv420p', crf=10)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    process2.stdin.write(
        cframe
        .astype(np.uint8)
        .tobytes()
    )

process2.stdin.close()
process2.wait()
