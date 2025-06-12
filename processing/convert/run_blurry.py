#!/usr/bin/env python3
import numpy as np
import os
from os import path
import sys
import glob
import shutil
import configargparse
from subprocess import check_call
import cv2
import dv
import os
from os import path
import sys
from tqdm import tqdm
from PIL import Image
from dvputils import getStreamByCameraId

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

def extract_frames(infn: str, t0: int, start: float, end: float, out_path: str):
    # return frame f_i such that i=argmax_i t_i<=timestamp
    # if the camera is exposing at timestamp, return that frame
    # if the camera is not exposing at timestamp, return last frame that did
    # extract frames starting with start: if exposing at start, return that, if not - return last exposed

    TIMESTAMP_PER_SEC = 1e6

    start *= TIMESTAMP_PER_SEC
    end *= TIMESTAMP_PER_SEC

    for camid in range(6):
        print(f'reading camera {camid}...')
        mcr = getStreamByCameraId(infn, camid)
        idx = 0
        while mcr.isRunning():
            frame = mcr.getNextFrame()
            ts = frame.timestamp  # start of exposure
            te = ts + frame.exposure.microseconds
            # print(ts-t0, timestamp)
            if te-t0 < start:
                continue
            # if ts-t0 > end:
            #     break
            print('good')
            print(ts-t0, te-t0, start, end)

            print(frame.image)
            img = np.tile(frame.image[..., None], (1, 1, 3))

            # os.makedirs(out_path, exist_ok=True)

            PERIOD = 0.200*TIMESTAMP_PER_SEC  # 5 fps
            # seq_timea = (round((ts-t0-start)/PERIOD)*PERIOD)/(end-start)*1000
            # seq_time_exact = (te-t0-start)/(end-start)*1000
            # seq_time = seq_timeb
            seq_time = (np.floor((te-t0-start)/PERIOD)*PERIOD)/(end-start)*1000
            ts_path = path.join(out_path, f'{int(seq_time):04d}')
            os.makedirs(ts_path, exist_ok=True)
            # out_fn = path.join(out_path, path.splitext(path.basename(infn))[0]+f'_{camid}_{idx}_{(frame.timestamp-t0)/TIMESTAMP_PER_SEC:.2f}.png')
            # out_fn = path.join(out_path, path.splitext(path.basename(infn))[0]+f'_{camid}_{idx}_{seq_time:.2f}.png')
            out_fn = path.join(ts_path, path.splitext(path.basename(infn))[0]+f'_{camid}_{idx}_{seq_time:.2f}.png')

            if seq_time > 1000:
                break

            Image.fromarray(img).save(out_fn)
            idx += 1


def process_calib(out_path, calib):
    intr = path.join(out_path, 'intrinsics')

    raw_pose = path.join(out_path, 'tmp', 'pose_raw')
    norm_pose = path.join(out_path, 'pose')
    norm_camdict = path.join(out_path, 'camdict.json')

    check_call(['./3_extract_params.py', calib, intr, raw_pose])
    check_call(['./4_normalize_poses.py', intr, raw_pose, norm_pose, norm_camdict])


def process_aedat(out_path, aedat, t0, start, end, bg_time, fps):
    event_dir = path.join(out_path, 'events')
    rgb_raw_dir = path.join(out_path, 'tmp', 'rgb_raw')
    background_raw_dir = path.join(out_path, 'tmp', 'background_raw')

    rgb_dir = path.join(out_path, 'rgb')
    background_dir = path.join(out_path, 'background')

    extract_frames(aedat, t0, start, end, rgb_raw_dir)
    # check_call(['./2_extract_mvframe.py', aedat, str(t0), str(start), rgb_raw_dir])
    # extract_frame_with_edi(aedat, t0, start, rgb_raw_dir)
    # extract_frames_with_edi(aedat, t0, start, end, fps, rgb_raw_dir)
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
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        out_path = f'{path.splitext(args.aedat)[0]}_{args.t0}_{args.start}_{args.end}'

    # process_calib(out_path, args.calib)
    process_aedat(out_path, args.aedat, args.t0, args.start, args.end, args.bg_time, args.fps)
