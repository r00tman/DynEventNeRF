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

parser.add_argument('-a', '--inputa', type=str, help='image sequence glob',
        default='../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t0-110_teps3e-2/render_circle_start_bt_124000/r_?????__*.png')

parser.add_argument('-b', '--inputb', type=str, help='image sequence glob',
        default='../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t090-210_teps3e-2/render_circle_middle_bt_136000/r_?????__*.png')

parser.add_argument('-o', '--output', type=str, help='output video name (leave empty for the basename)',
        default='tmp.mp4')

# parser.add_argument('-f', '--frame_list', type=str, help='frame list to render (renders all found frames if not specified)',
#         default=None)

# parser.add_argument('-r', '--relaxed', action='store_true', help='don\'t fail if not all frames of frame list are found',
#         default=None)

# parser.add_argument('-c', '--continuous', action='store_true', help='ignore the frame numbers, use each image once',
#         default=None)

# parser.add_argument('-R', '--frame_rate', type=int, help='frame rate of the video (default: 60 FPS)',
#         default=60)

# parser.add_argument('-l', '--loop_count', type=int, help='how many times to repeat the sequence (default: 1)',
#         default=1)

args = parser.parse_args()

filesa = glob(args.inputa)
filesb = glob(args.inputb)
filesa.sort()
filesb.sort()
files = []
files_d = dict()
for x in filesa+filesb:
    viewname = path.basename(x)
    if viewname not in files_d:
        files_d[viewname] = []
        files.append(viewname)
    files_d[viewname].append(x)

print(files)
print(files_d)

c_begin = None
c_end = None
for idx, viewname in enumerate(files):
    if len(files_d[viewname]) > 1 and c_begin is None:
        c_begin = idx-1
    if len(files_d[viewname]) == 1 and c_begin is not None and c_end is None:
        c_end = idx

if c_end is None and c_begin is not None:
    c_end = len(files)


print(c_begin, c_end)
print(files[c_begin], files[c_end])

# if args.output:
#     out_filename = args.output
# else:
#     out_filename = path.basename(path.dirname(input_glob))+f'_{args.frame_rate}fps{"_с" if args.continuous else ""}_{args.loop_count}x.mp4'
out_dir = '.'
out_filename = "test.mp4"


height, width = np.array(Image.open(files_d[files[0]][0])).shape[:2]

process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=30)
    .output(path.join(out_dir, out_filename), pix_fmt='yuv420p', crf=10)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

for idx, viewname in enumerate(files):

    print(viewname, files_d[viewname])
    cframes = np.stack([np.array(Image.open(x)) for x in files_d[viewname]], 0)
    cframes = cframes/255.
    cframes = cframes**2.2
    print(cframes.shape)
    # cframes = np.mean(cframes, 0)
    if cframes.shape[0] > 1:
        # c = 0.
        c = (idx-c_begin)/(c_end-c_begin)
        print(idx, c, c_begin, c_end)
        cframes = cframes[0]*(1-c)+cframes[1]*c
    print(cframes.shape)
    cframes = cframes**(1/2.2)
    cframes = (cframes*255.).astype(np.int8)

    process2.stdin.write(
        cframes
        .astype(np.uint8)
        .tobytes()
    )

process2.stdin.close()
process2.wait()
