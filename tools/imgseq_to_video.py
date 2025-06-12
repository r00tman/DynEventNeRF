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

parser.add_argument('-i', '--input', type=str, help='image sequence glob',
        default='scrap/testbt1/r_?????.png')

parser.add_argument('-o', '--output', type=str, help='output video name (leave empty for the basename)',
        default='')

parser.add_argument('-f', '--frame_list', type=str, help='frame list to render (renders all found frames if not specified)',
        default=None)

parser.add_argument('-r', '--relaxed', action='store_true', help='don\'t fail if not all frames of frame list are found',
        default=None)

parser.add_argument('-c', '--continuous', action='store_true', help='ignore the frame numbers, use each image once',
        default=None)

parser.add_argument('-R', '--frame_rate', type=int, help='frame rate of the video (default: 60 FPS)',
        default=60)

parser.add_argument('-l', '--loop_count', type=int, help='how many times to repeat the sequence (default: 1)',
        default=1)

args = parser.parse_args()

input_glob = args.input

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

a = glob(input_glob)
a.sort()

out_dir = path.join(path.dirname(input_glob), '..')
os.makedirs(out_dir, exist_ok=True)
if args.output:
    out_filename = args.output
else:
    out_filename = path.basename(path.dirname(input_glob))+f'_{args.frame_rate}fps{"_с" if args.continuous else ""}_{args.loop_count}x.mp4'


def find_png(x, suffix):
    res = glob(path.join(x, suffix))
    res.sort()
    if len(res) == 0:
        return None
    return res[-1]

c = a

def extract_frame_id(x):
    # s = re.search(r'frame_([0-9]*)', x)
    # s = re.search(r'_([0-9][0-9]*).png$', x)
    s = re.search(r'_*([0-9][0-9]*).png$', x)
    if s is None:
        return None
    else:
        return int(s.group(1))

print(c)
c = [x for x in c if extract_frame_id(x) is not None]
print([(extract_frame_id(x),x) for x in c])
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


d = [(extract_frame_id(x), x) for x in c]
d = [x for x in d if x[1] is not None]

print(d)

height, width = np.array(Image.open(d[0][1])).shape[:2]

process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=args.frame_rate)
    .output(path.join(out_dir, out_filename), pix_fmt='yuv420p', crf=10)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

for _ in range(args.loop_count):
    if args.continuous:
        frame_id = 0
        while frame_id < len(d):
            cframe = np.array(Image.open(d[frame_id][1]))
            if len(cframe.shape) < 3:
                cframe = np.tile(cframe[..., None], (1, 1, 3))

            process2.stdin.write(
                cframe
                .astype(np.uint8)
                .tobytes()
            )
            frame_id += 1
    else:
        frame_id = 0
        index = 0
        cframe = None
        while frame_id <= 1000:
            while d[index][0] < frame_id:
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
