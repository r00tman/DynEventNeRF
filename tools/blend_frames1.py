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

parser.add_argument('-i', '--inputs', nargs='+', type=str, help='image sequence glob',
        default=[
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t0-110_teps3e-2/render_circle_start_bt_124000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t090-210_teps3e-2/render_circle_middle_bt_136000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t190-310_teps3e-2/render_circle_middle_bt_124000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t290-410_teps3e-2/render_circle_middle_bt_120000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t390-510_teps3e-2/render_circle_middle_bt_124000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t490-610_teps3e-2/render_circle_middle_bt_130000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t590-710_teps3e-2/render_circle_middle_bt_120000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t690-810_teps3e-2/render_circle_middle_bt_120000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t790-910_teps3e-2/render_circle_middle_bt_122000/r_?????__*.png',
        # '../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t890-1000_teps3e-2/render_circle_end_bt_120000/r_?????__*.png',

        # '../logs_auto/mlp_seg_59_64_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t0-220_teps3e-2/render_circle_start_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_59_64_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t180-420_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_59_64_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t380-620_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_59_64_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t580-820_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_59_64_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t780-1000_teps3e-2/render_circle_end_bt_1?????/r_?????__*.png',

        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t000-110_teps3e-2/render_circle_start_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t090-210_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t190-310_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t290-410_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_nolrsched_t390-510_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',

        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t0-110_teps3e-2/render_circle_start_bt_03????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t90-210_teps3e-2/render_circle_middle_bt_03????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t190-310_teps3e-2/render_circle_middle_bt_03????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t290-410_teps3e-2/render_circle_middle_bt_03????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t390-510_teps3e-2/render_circle_middle_bt_03????/r_?????__*.png',

        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t0-110_teps3e-2/render_circle_start_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t90-210_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t190-310_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t290-410_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_370_380_5fps_ls0.5_3e-2_newcode_lambdaanneal10k_1e-2_nolrsched_t390-510_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',

        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t0-110_teps3e-2/render_circle_start_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t90-210_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t290-410_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t390-510_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t490-610_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t590-710_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t690-810_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t790-910_teps3e-2/render_circle_middle_bt_1?????/r_?????__*.png',
        # '../logs_auto/mlp_seg_146_156_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t890-1000_teps3e-2/render_circle_end_bt_1?????/r_?????__*.png',

        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t0-110_teps3e-2/render_circle_start_bt_094000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t90-210_teps3e-2/render_circle_middle_bt_100000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2/render_circle_middle_bt_106000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t290-410_teps3e-2/render_circle_middle_bt_102000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t390-510_teps3e-2/render_circle_middle_bt_094000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t490-610_teps3e-2/render_circle_middle_bt_102000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t590-710_teps3e-2/render_circle_middle_bt_096000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t690-810_teps3e-2/render_circle_middle_bt_098000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t790-910_teps3e-2/render_circle_middle_bt_094000/r_?????_fg_*.png',
        # '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t890-1000_teps3e-2/render_circle_end_bt_02????/r_?????__*.png',

        # '../logs_auto/cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_0_210/render_test_5view_120000/????????????????????????__????.png',
        # '../logs_auto/cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_190_410/render_test_5view_118000/????????????????????????__????.png',
        # '../logs_auto/cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_390_610/render_test_5view_120000/????????????????????????__????.png',
        # '../logs_auto/cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_590_810/render_test_5view_120000/????????????????????????__????.png',
        # '../logs_auto/cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_790_1000/render_test_5view_122000/????????????????????????__????.png',

        '../logs_auto/cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_0_210/render_test_5view_150000/????????????????????????__????.png',
        '../logs_auto/cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_190_410/render_test_5view_150000/????????????????????????__????.png',
        '../logs_auto/cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_390_610/render_test_5view_150000/????????????????????????__????.png',
        '../logs_auto/cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_590_810/render_test_5view_150000/????????????????????????__????.png',
        '../logs_auto/cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_790_1000/render_test_5view_150000/????????????????????????__????.png',
     ])

# parser.add_argument('-a', '--inputa', type=str, help='image sequence glob',
#         default='../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t0-110_teps3e-2/render_circle_start_bt_124000/r_?????__*.png')

# parser.add_argument('-b', '--inputb', type=str, help='image sequence glob',
#         default='../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t090-210_teps3e-2/render_circle_middle_bt_136000/r_?????__*.png')

# parser.add_argument('-c', '--inputc', type=str, help='image sequence glob',
#         default='../logs_auto/mlp_seg_292_302_5fps_linearshift0.6_3e-2_newcode_nolambdaanneal_nolrsched_t190-310_teps3e-2/render_circle_middle_bt_124000/r_?????__*.png')

parser.add_argument('-o', '--output', type=str, help='output video name (leave empty for the basename)',
        default='test2904_100_105_2_fg.mp4')

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

all_files = [list(sorted(glob(x))) for x in args.inputs]
for x, y in zip(args.inputs, all_files):
    if len(y) == 0:
        assert False, f'there are no files in {x}'
all_files = sum(all_files, [])
# all_files = [x for x in all_files if 'depth' not in x]
# filesa = glob(args.inputa)
# filesb = glob(args.inputb)
# filesc = glob(args.inputc)
# filesa.sort()
# filesb.sort()
# filesc.sort()
files = []
files_d = dict()
# for x in filesa+filesb+filesc:
for x in all_files:
    viewname = path.basename(x)
    if viewname not in files_d:
        files_d[viewname] = []
        files.append(viewname)
    files_d[viewname].append(x)

print(files)
print(files_d)



# if args.output:
#     out_filename = args.output
# else:
#     out_filename = path.basename(path.dirname(input_glob))+f'_{args.frame_rate}fps{"_с" if args.continuous else ""}_{args.loop_count}x.mp4'
out_dir = '.'
out_filename = args.output


height, width = np.array(Image.open(files_d[files[0]][0])).shape[:2]

process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=30)
    .output(path.join(out_dir, out_filename), pix_fmt='yuv420p', crf=10)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

c_begin = None
c_end = None
for idx, viewname in enumerate(files):

    print(viewname, files_d[viewname])
    cframes = np.stack([np.array(Image.open(x)) for x in files_d[viewname]], 0)
    print(cframes.shape)
    # cframes = np.mean(cframes, 0)
    if cframes.shape[0] > 1:
        if c_begin is None:
            # breakpoint()
            for j, vn in list(enumerate(files))[idx:]:
                if len(files_d[vn]) > 1 and c_begin is None:
                    c_begin = j-1
                if len(files_d[vn]) == 1 and c_begin is not None and c_end is None:
                    c_end = j
                    break

            if c_end is None and c_begin is not None:
                c_end = len(files)-1

        print('found', c_begin, c_end)
        print(files[c_begin], files[c_end])
        # c = 0.
        c = (idx-c_begin)/(c_end-c_begin)
        print(idx, c, c_begin, c_end)
        cframes = cframes/255.
        cframes = cframes**2.2
        cframes = cframes[0]*(1-c)+cframes[1]*c
        cframes = cframes**(1/2.2)
        cframes = (cframes*255.).astype(np.int8)

    if c_end is not None and idx >= c_end:
        c_begin = None
        c_end = None

    print(cframes.shape)

    process2.stdin.write(
        cframes
        .astype(np.uint8)
        .tobytes()
    )

process2.stdin.close()
process2.wait()
