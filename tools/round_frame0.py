#!/usr/bin/env python3
import numpy as np
from PIL import Image
import cv2
import sys
import os
import shutil
from os import path
from glob import glob
from tqdm import tqdm
from debayer import debayer_image


if len(sys.argv) < 3:
    print(
f'''Usage: {sys.argv[0]} <T0_RENDER_PATH> <OUT_DIR> [DEBAYER] [BG_COLOR] [H] [W]

Rounds up raw RGB frame to bayered event grid for better accumulation


Example: ./round_frame0.py '../../singleframe/train/rgb/r_??_0000.png' rgb''')
    exit()

T0_FRAME_ID = 0
T0_RENDER_PATH = sys.argv[1]
OUT_DIR = sys.argv[2]
DEBAYER = False
if len(sys.argv) > 3:
    DEBAYER = eval(sys.argv[3])

THR = 0.5  # event generation threshold
BG_VAL = 159  # background color
if len(sys.argv) > 4:
    BG_VAL = int(sys.argv[4])

# W, H = 346, 260
W, H = None, None

EVENT_EPS = 1e-6  # event generation eps
TONEMAP_EPS = 0.01  # tonemapping eps (as in np.log(...+TONEMAP_EPS))

print('Arguments:')
print(f'T0_RENDER_PATH = "{T0_RENDER_PATH}"')
print(f'OUT_DIR = "{OUT_DIR}"')
print(f'W, H = {W}, {H}')
print(f'THR = {THR}')
print(f'BG_VAL = {BG_VAL}')
print(f'EVENT_EPS = {EVENT_EPS}')
print(f'TONEMAP_EPS = {TONEMAP_EPS}')
print(f'DEBAYER = {DEBAYER}')
print()


def full_split_path(p):
    # c/a/????/b/????/???.png -> ['c', 'a', '????', 'b', '????', '???']
    res = []

    while True:
        p, f = path.split(p)

        if f != '':
            res.append(f)
        else:
            if p != '':
                res.append(p)
            break

    res.reverse()
    return res

def get_view_name(fn, render_mask):
    # c/a/????/b/????/???.png -> ????/b/????/???

    fn_parts = full_split_path(fn)
    mask_parts = full_split_path(render_mask)

    min_glob = None
    for i, p in enumerate(mask_parts):
        if '?' in p or '*' in p:
            min_glob = i
            break

    if min_glob is not None:
        res = path.join(*fn_parts[min_glob:])
    else:
        res = fn

    res = path.splitext(res)[0]
    return res

def list_views(p):
    res = glob(p, recursive=True)
    def is_image(fn: str):
        return fn.lower().endswith(('.png', '.jpg'))
    res = list(filter(is_image, res))
    res.sort()
    return res


def srgb_to_log(a):
    '''
    a: 0-255 [H, W]
    turns 0-255 srgb into 0-1 abs then to log brightness
    '''
    a = a/255
    linear = a ** 2.2
    log = np.log(linear+TONEMAP_EPS)
    return log


def log_to_srgb(log):
    '''
    log: 0-255 [H, W]
    turns log brightness into 0-1 abs then to 0-255 srgb
    '''
    linear = np.exp(log)-TONEMAP_EPS
    a = np.maximum(0, linear) ** (1 / 2.2)
    a = a*255
    return a


def stats(x):
    print(x.dtype, x.shape, x.min(), x.max(), np.quantile(x, 0.25), np.median(x), np.quantile(x, 0.75), x.mean(), sep='\t')


print(f'Recreating output folder: "{OUT_DIR}"...')
# try:
#     shutil.rmtree(OUT_DIR)
# except FileNotFoundError:
#     pass


os.makedirs(OUT_DIR, exist_ok=True)

views_t0_render = list_views(T0_RENDER_PATH)

print('Found t0 render views:')
print(*views_t0_render, sep='\n')
print()



print('Rendering...')
out_fns = []
for t0render_fn in tqdm(views_t0_render):
    t0render = np.array(Image.open(t0render_fn)).astype(np.int16)[..., :3]

    if H is None:
        H, W = t0render.shape[:2]
        print(f'Detected resolution: {W}x{H}')
        # Prepare Bayer RGGB filter
        color_mask = np.zeros((H, W, 3))

        color_mask[0::2, 0::2, 0] = 1  # r

        color_mask[0::2, 1::2, 1] = 1  # g
        color_mask[1::2, 0::2, 1] = 1  # g

        color_mask[1::2, 1::2, 2] = 1  # b

    # take care of RGGB Bayer masks
    t0render_mono = (t0render * color_mask).sum(-1)
    assert len(t0render_mono.shape) == 2  # h*w
    assert t0render_mono.shape == (H, W)  # h*w

    # oh, it's not so easy, one needs proper tone-mapping
    # so, t0render_mono needs to go from srgb to abs and then to log
    # then we sum stuff up with the proper threshold
    # and then we do everything in reverse: log to abs to srgb
    base = np.log((BG_VAL/255.)**2.2+TONEMAP_EPS)

    def round_evts(x):
        sgn = np.sign(x)
        # event is reached only when full THR is passed, therefore we take floor of the abs number of events passed
        # if we just divide x//THR, then -0.6//0.5=-2, not -1 as we want
        # test cases:
        # 0.6, THR=0.5 -> 1
        # -0.6, THR=0.5 -> -1
        # POSSIBLY missing handling of EPS=1e-6
        #
        # cnt = np.floor(np.abs(x)//THR) <-- old one
        # cnt = np.floor(np.abs(x)//(THR-1e-6)) <-- incorrect as only the last event is affected by eps. eps is used originally to push the events when THR-eps passed rather than THR. but even then mem_frame was increased by the full THR, not THR-eps.
        # this one should do the correct handling
        # the gt 0 to gt 1000 example results in the same images, regardless of whether i use eps or no. so it's hard to confirm correctness experimentally
        cnt = np.floor((np.abs(x)+EVENT_EPS)/THR)
        return cnt*sgn*THR

    out = log_to_srgb(round_evts(srgb_to_log(t0render_mono)-base)+base)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if DEBAYER:
        out = debayer_image(out)

    # view_name = path.splitext(path.basename(t0render_fn))[0]
    view_name = get_view_name(t0render_fn, T0_RENDER_PATH)

    if False and f'{T0_FRAME_ID:04d}' in view_name:
        out_fn = path.join(OUT_DIR, view_name.replace(f'{T0_FRAME_ID:04d}', f'{T0_FRAME_ID:04d}')+'.png')
    else:
        # out_fn = path.join(OUT_DIR, view_name+f'_synth{T0_FRAME_ID:04d}.png')
        out_fn = path.join(OUT_DIR, view_name+f'_from{T0_FRAME_ID:04d}_synth{T0_FRAME_ID:04d}.png')
    os.makedirs(path.dirname(out_fn), exist_ok=True)
    Image.fromarray(out).save(out_fn)
    out_fns.append(out_fn)

print()
print('Rendered following views:')
print(*out_fns, sep='\n')
