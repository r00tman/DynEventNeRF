#!/usr/bin/env python3
import numpy as np
import os
from os import path
import sys
import glob
import shutil
import tqdm
import configargparse
import ffmpeg
import cv2
from subprocess import check_call
import dvpstat_cpp
import dv_processing as dvp
from dvputils import getStreamByCameraId
from concurrent.futures import ThreadPoolExecutor

# steps:
# 1. extractbgandinit

# can be separated as:
# - poses
#  - 3. convertcalib
#  - 4. normalizeposes
#
# - events and images
#  - 1. aedat4tonpz
#  - 2. extractbgandinit
#  - 5. debayer

EDI_CMD = [
    './edi3',
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
    '-p', '0.32',
    '-n', '-0.32',
    '-e', '3e-2',
    '-s', 'linearshift',

    # linearshift 0.5
    # '-p', '0.5',
    # '-n', '-0.5',
    # '-e', '3e-2',
    # '-s', 'linearshift',

    '--debayering', 'none'
]


def linear_to_srgb(linear):
    '''
    linear: 0-255 [H, W]
    turns log brightness into 0-1 abs then to 0-255 srgb
    '''
    linear = linear / 255.
    a = np.maximum(0, linear) ** (1 / 2.2)
    a = np.clip(a*255, 0, 255).astype(np.uint8)
    return a


def debayer_srgb(a):
    if a.shape[-1] == 3:
        a = a[..., 0]

    default = cv2.cvtColor(a, cv2.COLOR_BayerBG2RGB)
    vng = cv2.cvtColor(a, cv2.COLOR_BayerBG2RGB_VNG)

    # correct vng y=-2 shift
    result = np.copy(default)
    result[2:] = vng[:-2]

    result = linear_to_srgb(result)

    return result

def debayer_srgb_video(inp, out, rate=None):
    probe = ffmpeg.probe(inp)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    if rate is None:
        rate = video_stream['r_frame_rate']
    # rate = 1

    process1 = (
        ffmpeg
        .input(inp)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(2*width, 2*height), r=rate)
        .output(path.join(out), pix_fmt='yuv420p', crf=10)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )

        out_frame = debayer_srgb(in_frame)
        out_frame = cv2.resize(out_frame, dsize=(width*2, height*2), interpolation=cv2.INTER_CUBIC)

        process2.stdin.write(
            out_frame
            .astype(np.uint8)
            .tobytes()
        )

    process2.stdin.close()
    process1.wait()
    process2.wait()




def extract_frames_with_edi(aedat, t0, start, end, fps, out_path, cam_idx):
    video_dir = out_path+'_video'
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # # offset to 0
    # t0 -= (timestamp + 1) * 1e6
    # timestamp = 1.

    # assert(start - 1 > 0)
    out_video = path.join(video_dir, f'{cam_idx}.mp4')
    out_video_proc = path.join(out_path, f'{cam_idx}.mp4')
    check_call(EDI_CMD + ['-a', str(start),
                          '-b', str(end),
                          f'--t0={t0}',
                          '-r', str(fps),
                         '-c', str(cam_idx),
                          '-o', out_video,
                          '-i', aedat])

    debayer_srgb_video(out_video, out_video_proc, fps)


def detect_aedatparams(aedat, t0, start, end):
    lowesttimes = []
    highesttimes = []
    for camid in range(6):
        with tqdm.tqdm() as pbar:
            mcr = getStreamByCameraId(aedat, camid)
            a = mcr.getNextEventBatch()

            lowesttimes.append(a.getLowestTime())
            highesttime = a.getHighestTime()
            lastupd = highesttime

            if end >= 0:
                highesttime = end*1e-6+t0
            else:
                while mcr.isRunning():
                    a = mcr.getNextEventBatch()
                    if a is None:
                        break
                    highesttime = max(highesttime, a.getHighestTime())
                    pbar.update((highesttime-lastupd)/1e6)
                    lastupd = highesttime
            highesttimes.append(highesttime)


    print(lowesttimes)
    print(highesttimes)

    if t0 < 0:
        t0 = min(*lowesttimes)
    if start < 0:
        start = 0
    if end < 0:
        end = (max(*highesttimes)-t0)//1e6

    return t0, start, end




def process_aedat(out_path, aedat, t0, start, end, fps):
    rgb_raw_dir = path.join(out_path, 'rgb_raw')
    rgb_dir = path.join(out_path, 'rgb')

    if t0 < 0 or start < 0 or end < 0:
        print('detecting...')
        t0, start, end = detect_aedatparams(aedat, t0, start, end)
        print(f'detected t0={t0} start={start} end={end}')

    print('extracting')

    # Define the number of threads you want in your pool
    num_threads = 3

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        func = lambda cam_idx: extract_frames_with_edi(aedat, t0, start, end, fps, rgb_raw_dir, cam_idx)
        camindices = range(6)
        futures = [executor.submit(func, cam_idx) for cam_idx in camindices]

        # Get the results from the futures
        results = [future.result() for future in futures]

    # for cam_idx in range(6):
    #     extract_frames_with_edi(aedat, t0, start, end, fps, rgb_raw_dir, cam_idx)


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', type=str, help="output path",
                        default=None)
    parser.add_argument('-a', '--aedat', type=str, help="event+frames multi-view aedat4 file",
                        default=None)
    parser.add_argument('--t0', type=int, help="common t0 timestamp",
                        default=-1)
    parser.add_argument('-s', '--start', type=float, help="start time in seconds",
                        default=-1)
    parser.add_argument('-e', '--end', type=float, help="end time in seconds",
                        default=-1)
    parser.add_argument('-r', '--fps', type=float, help="rgb fps",
                        default=1)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()


    out_path = args.out_path
    if out_path is None:
        out_path = f'{path.splitext(args.aedat)[0]}_{args.t0}_{args.start}_{args.end}'

    process_aedat(out_path, args.aedat, args.t0, args.start, args.end, args.fps)
