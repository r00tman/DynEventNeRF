#!/usr/bin/env python3
import os
from os import path
from glob import glob
import re

import numpy as np
import ffmpeg
from PIL import Image
import configargparse

parser = configargparse.ArgumentParser()

parser.add_argument('-w', '--work_dir', type=str, help='predictions directory',
        default='logs_auto/auto_slurm')

parser.add_argument('-o', '--out_suffix', type=str, help='output videos directory suffix',
        default='videos')

parser.add_argument('-f', '--frame_list', type=str, help='frame list to render (renders all found frames if not specified)',
        default=None)

parser.add_argument('-r', '--relaxed', action='store_true', help='don\'t fail if not all frames of frame list are found',
        default=None)

parser.add_argument('-n', '--frame_count', type=int, help='total number of frames',
        default=1000)

args = parser.parse_args()

# b = \
# '''train_frame_0000
# train_frame_0500
# train_frame_0750
# train_frame_1000
# '''.splitlines()

# b = [path.join('logs_auto', x) for x in b]

# a = glob('logs_auto/*interactive1')
# work_dir = 'logs_auto/trfcp_bt'
# work_dir = 'logs_auto/trfcp_bt1'
# work_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs_auto/auto_slurm'
work_dir = args.work_dir

# frame_list = sys.argv[2] if len(sys.argv) > 2 else None
frame_list = args.frame_list
# frame_list = 'frame_lists/1_frames.txt'
# frame_list = 'frame_lists/17_frames.txt'
if frame_list:
    print('using frame list:', frame_list)
    with open(frame_list) as f:
        frame_list = [int(l.strip()) for l in f.readlines() if l.strip()]
    frame_list = set(frame_list)
    print('found', len(frame_list), 'frames', 'in the list')

a = glob(path.join(work_dir, '*'))
# out_dir = path.join(work_dir, 'videos')
out_dir = path.join(work_dir, args.out_suffix)
os.makedirs(out_dir, exist_ok=True)

suffixes = [
    ('test00050.mp4', 'render_validation_*/r_00050.png'),
    ('test00383.mp4', 'render_validation_*/r_00383.png'),
    ('test00716.mp4', 'render_validation_*/r_00716.png'),

    ('train00.mp4', 'render_train_*/r_00_0000.png'),
    ('train01.mp4', 'render_train_*/r_01_0000.png'),
    ('train02.mp4', 'render_train_*/r_02_0000.png'),
    ('train03.mp4', 'render_train_*/r_03_0000.png'),
    ('train04.mp4', 'render_train_*/r_04_0000.png'),

    ('test00050_depth.mp4', 'render_validation_*/fg_depth_r_00050.png'),
    ('test00383_depth.mp4', 'render_validation_*/fg_depth_r_00383.png'),
    ('test00716_depth.mp4', 'render_validation_*/fg_depth_r_00716.png'),

    ('train00_depth.mp4', 'render_train_*/fg_depth_r_00_0000.png'),
    ('train01_depth.mp4', 'render_train_*/fg_depth_r_01_0000.png'),
    ('train02_depth.mp4', 'render_train_*/fg_depth_r_02_0000.png'),
    ('train03_depth.mp4', 'render_train_*/fg_depth_r_03_0000.png'),
    ('train04_depth.mp4', 'render_train_*/fg_depth_r_04_0000.png'),

    ('test00050_fg.mp4', 'render_validation_*/fg_r_00050.png'),
    ('test00383_fg.mp4', 'render_validation_*/fg_r_00383.png'),
    ('test00716_fg.mp4', 'render_validation_*/fg_r_00716.png'),

    ('train00_fg.mp4', 'render_train_*/fg_r_00_0000.png'),
    ('train01_fg.mp4', 'render_train_*/fg_r_01_0000.png'),
    ('train02_fg.mp4', 'render_train_*/fg_r_02_0000.png'),
    ('train03_fg.mp4', 'render_train_*/fg_r_03_0000.png'),
    ('train04_fg.mp4', 'render_train_*/fg_r_04_0000.png'),
]


def find_png(x, suffix):
    res = glob(path.join(x, suffix))
    res.sort()
    if len(res) == 0:
        return None
    return res[-1]

# c = a+b
c = a
def extract_frame_id(x):
    # s = re.search(r'frame_([0-9]*)', x)
    s = re.search(r'train_([0-9]*)', x)
    if s is None:
        return None
    else:
        return int(s.group(1))

print(c)
c = [x for x in c if extract_frame_id(x) is not None]
if frame_list:
    c = [x for x in c if extract_frame_id(x) in frame_list]
    if args.relaxed:
        if len(c) != len(frame_list):
            print(f'WARNING: all {len(frame_list)} frames should be there, now there are only {len(c)} frames')
    else:
        assert len(c) == len(frame_list), f'all {len(frame_list)} frames must be there, now there are only {len(c)} frames'

print(c)
c.sort(key=extract_frame_id)
print(c)


for out_filename, suffix in suffixes:
    d = [(extract_frame_id(x), find_png(x, suffix)) for x in c]
    d = [x for x in d if x[1] is not None]

    print(d)

    height, width, _ = np.array(Image.open(d[0][1])).shape

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=60)
        .output(path.join(out_dir, out_filename), pix_fmt='yuv420p', crf=10)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    frame_id = 0
    index = 0
    cframe = None
    while frame_id <= args.frame_count:
        # it should be 0 0 0 0 4 4 4 4 8 8 8 8
        # only when frame_id reaches next d, then increment
        while index+1 < len(d) and d[index+1][0] < frame_id:
            index += 1
            cframe = None
        if cframe is None:
            print(index, d[index])
            cframe = np.array(Image.open(d[index][1]))

        process2.stdin.write(
            cframe
            .astype(np.uint8)
            .tobytes()
        )
        frame_id += 1

    process2.stdin.close()
    process2.wait()
