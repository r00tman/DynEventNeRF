#!/usr/bin/env python3
import numpy as np
import os
from os import path
import sys
import glob
import shutil
import configargparse
from subprocess import check_call

# steps:
# 1. aedat4tonpz
# 2. extractbgandinit
# 3. convertcalib
# 4. normalizeposes
# 5. debayer

# can be separated as:
# - poses
#  - 3. convertcalib
#  - 4. normalizeposes
#
# - events and images
#  - 1. aedat4tonpz
#  - 2. extractbgandinit
#  - 5. debayer

def process_calib(out_path, calib):
    intr = path.join(out_path, 'intrinsics')

    raw_pose = path.join(out_path, 'tmp', 'pose_raw')
    norm_pose = path.join(out_path, 'pose')
    norm_camdict = path.join(out_path, 'camdict.json')

    check_call(['./3_extract_params.py', calib, intr, raw_pose])
    check_call(['./4_normalize_poses.py', intr, raw_pose, norm_pose, norm_camdict])


EDI_CMD = [
    '../edi/edi3',
    # values from the new studio data
    # '-p', '0.3',
    # '-n', '-0.5',
    # '-e', '0e-3',
    # '-s', 'linear',

    # values from the new studio data (first line)
    # '-p', '0.7',
    # '-n', '-0.7',
    # '-e', '12e-3',
    # '-s', 'srgb',

    # linear 0.6
    # '-p', '0.6',
    # '-n', '-0.6',
    # '-e', '0',
    # '-s', 'linear',

    # linear 0.32
    # '-p', '0.32',
    # '-n', '-0.32',
    # '-e', '0',
    # '-s', 'linear',

    # linearshift 0.6
    # '-p', '0.6',
    # '-n', '-0.6',
    # '-e', '3e-2',
    # '-s', 'linearshift',

    # linearshift 0.32
    # '-p', '0.32',
    # '-n', '-0.32',
    # '-e', '3e-2',
    # '-s', 'linearshift',

    # linearshift 0.5
    '-p', '0.5',
    '-n', '-0.5',
    '-e', '3e-2',
    '-s', 'linearshift',

    '--debayering', 'none'
]


def extract_frame_with_edi(aedat, t0, timestamp, out_path):
    video_dir = out_path+'_video'
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # # offset to 0
    # t0 -= (timestamp + 1) * 1e6
    # timestamp = 1.

    assert(timestamp - 1 > 0)
    for camIdx in range(6):
        out_video = path.join(video_dir, f'{camIdx}.mp4')
        check_call(EDI_CMD + ['-a', str(timestamp-1),
                              '-b', str(timestamp+1),
                              f'--t0={t0}',
                              '-r', '1',
                              '-c', str(camIdx),
                              '-o', out_video,
                              '-i', aedat])
        out_img = path.splitext(out_video)[0]+'_%02d.png'
        check_call(['ffmpeg', '-i', out_video, out_img])
        # frame 1 is black, frame 2 is what we need
        shutil.copyfile(out_img % 2, path.join(out_path, f'{t0}_{timestamp}_{camIdx}.png'))


def extract_frames_with_edi(aedat, t0, start, end, fps, out_path):
    video_dir = out_path+'_video'
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # # offset to 0
    # t0 -= (timestamp + 1) * 1e6
    # timestamp = 1.

    Tmax = 1000

    assert(start - 1 > 0)
    for camIdx in range(6):
        out_video = path.join(video_dir, f'{camIdx}.mp4')
        check_call(EDI_CMD + ['-a', str(start-1),
                              '-b', str(end+1),
                              f'--t0={t0}',
                              '-r', str(fps),
                             '-c', str(camIdx),
                              '-o', out_video,
                              '-i', aedat])
        out_img = path.splitext(out_video)[0]+'_%05d.png'
        check_call(['ffmpeg', '-i', out_video, out_img])
        # frame 1 is black, frame 2 is what we need
        # for f in range((end-start)*fps+1):
        print(glob.glob(out_img.replace('%05d', '?????')))
        for idx, fn in enumerate(sorted(glob.glob(out_img.replace('%05d', '?????')))):
            timestamp = start-1+idx/fps
            # T = int((timestamp-start)*Tmax/(end-start))
            T = int(np.round((timestamp-start)*Tmax/(end-start)))
            if 0 <= T <= Tmax:
                os.makedirs(path.join(out_path, f'{T:04d}'), exist_ok=True)
                shutil.copyfile(fn, path.join(out_path, f'{T:04d}', f'{t0}_{timestamp}_{camIdx}.png'))


def process_aedat(out_path, aedat, t0, start, end, bg_time, fps):
    event_dir = path.join(out_path, 'events')
    rgb_raw_dir = path.join(out_path, 'tmp', 'rgb_raw')
    background_raw_dir = path.join(out_path, 'tmp', 'background_raw')

    rgb_dir = path.join(out_path, 'rgb')
    background_dir = path.join(out_path, 'background')

    check_call(['./1_mv_aedat4tonpz.py', aedat, str(t0), str(start), str(end), event_dir])

    # check_call(['./2_extract_mvframe.py', aedat, str(t0), str(start), rgb_raw_dir])
    # extract_frame_with_edi(aedat, t0, start, rgb_raw_dir)
    extract_frames_with_edi(aedat, t0, start, end, fps, rgb_raw_dir)
    check_call(['./2_extract_mvframe.py', aedat, str(t0), str(bg_time), background_raw_dir])

    check_call(['./5_debayer_srgb.py', rgb_raw_dir, rgb_dir])
    check_call(['./5_debayer_srgb.py', background_raw_dir, background_dir])


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', type=str, help="output path",
                        default='output/sec59_64')
    parser.add_argument('-c', '--calib', type=str, help="captury calib file",
                        default='input/cameras_event.calib')
    parser.add_argument('-a', '--aedat', type=str, help="event+frames multi-view aedat4 file",
                        default='../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4')
    parser.add_argument('--t0', type=int, help="common t0 timestamp",
                        default=1705599197839417)
    parser.add_argument('-s', '--start', type=float, help="start time in seconds",
                        default=59)
    parser.add_argument('-e', '--end', type=float, help="end time in seconds",
                        default=64)
    parser.add_argument('-b', '--bg_time', type=float, help="time in seconds when to take the background",
                        default=109)
    parser.add_argument('-r', '--fps', type=float, help="rgb fps",
                        default=1)
    parser.add_argument('--no_calib', action='store_true', help="don't do calibration",
                        default=False)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        out_path = f'{path.splitext(args.aedat)[0]}_{args.t0}_{args.start}_{args.end}'

    if not args.no_calib:
        process_calib(out_path, args.calib)
    process_aedat(out_path, args.aedat, args.t0, args.start, args.end, args.bg_time, args.fps)
