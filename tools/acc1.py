#!/usr/bin/env python3
import numpy as np
from PIL import Image
import cv2
import sys
import os
import shutil
from os import path
import numba
from glob import glob
from tqdm import tqdm


if len(sys.argv) < 7:
    print(
f'''Usage: {sys.argv[0]} <MODE> <GT_EVENTS_PATH> <T0_RENDER_PATH> <T0_FRAME_ID> <T1_FRAME_ID> <OUT_DIR>

Accumulates multi-view events from t0 to t1 onto t0 renders.

MODE can be either:
 - 'accurate' -- round up t0 to the event grid using the background color. This results in exactly correct simulated gt 0 -> gt 1000 accumulation
 - 'direct' -- don't rount up t0 to the event grid. This might produce less artifacts with real t0 rendering inputs rather than gt t0.

Example: ./acc1.py 'accurate' 'events/*' '../../singleframe/train/rgb/r_??_0000.png' 0 1000 rgb''')
    exit()

MODE = sys.argv[1]
assert MODE in ['accurate', 'direct']
GT_EVENTS_PATH = sys.argv[2]
T0_RENDER_PATH = sys.argv[3]
T0_FRAME_ID = int(sys.argv[4])
T1_FRAME_ID = int(sys.argv[5])
OUT_DIR = sys.argv[6]

W, H = 346, 260
THR = 0.5  # event generation threshold
BG_VAL = 159  # background color
EVENT_EPS = 1e-6  # event generation eps
TONEMAP_EPS = 0.01  # tonemapping eps (as in np.log(...+TONEMAP_EPS))

print('Arguments:')
print(f'MODE = "{MODE}"')
print(f'GT_EVENTS_PATH = "{GT_EVENTS_PATH}"')
print(f'T0_RENDER_PATH = "{T0_RENDER_PATH}"')
print(f'T0_FRAME_ID = {T0_FRAME_ID}')
print(f'T1_FRAME_ID = {T1_FRAME_ID}')
print(f'OUT_DIR = "{OUT_DIR}"')
print(f'W, H = {W}, {H}')
print(f'THR = {THR}')
print(f'BG_VAL = {BG_VAL}')
print(f'EVENT_EPS = {EVENT_EPS}')
print(f'TONEMAP_EPS = {TONEMAP_EPS}')
print()


def list_views(p):
    res = glob(p)
    def is_image(fn: str):
        return fn.lower().endswith(('.png', '.jpg'))
    res = list(filter(is_image, res))
    res.sort()
    return res


def list_events(p):
    res = glob(p)
    def is_events(fn: str):
        return fn.lower().endswith('.npz')
    res = list(filter(is_events, res))
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


@numba.jit(nopython=True)
def accumulate(xs, ys, ts, ps, out):
    for i in range(len(xs)):
        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        out[y, x] += p


def stats(x):
    print(x.dtype, x.shape, x.min(), x.max(), np.quantile(x, 0.25), np.median(x), np.quantile(x, 0.75), x.mean(), sep='\t')


print(f'Recreating output folder: "{OUT_DIR}"...')
# try:
#     shutil.rmtree(OUT_DIR)
# except FileNotFoundError:
#     pass


os.makedirs(OUT_DIR, exist_ok=True)

views_t0_render = list_views(T0_RENDER_PATH)
views_events = list_events(GT_EVENTS_PATH)

print('Found t0 render views:')
print(*views_t0_render, sep='\n')
print()

print('Found event views:')
print(*views_events, sep='\n')
print()

assert len(views_t0_render) == len(views_events), 'Views must match between RGB renders and events'


# Prepare Bayer RGGB filter
color_mask = np.zeros((H, W, 3))

color_mask[0::2, 0::2, 0] = 1  # r

color_mask[0::2, 1::2, 1] = 1  # g
color_mask[1::2, 0::2, 1] = 1  # g

color_mask[1::2, 1::2, 2] = 1  # b


print('Rendering...')
out_fns = []
for t0render_fn, events_fn in tqdm(list(zip(views_t0_render, views_events))):
    # t0render_fn, events_fn = views_t0_render[0], views_events[0]

    t0render = np.array(Image.open(t0render_fn)).astype(np.int16)
    if len(t0render.shape) == 2:
        t0render = np.tile(t0render[..., None], (1, 1, 3))
    t0render = t0render[..., :3]
    assert len(t0render.shape) == 3

    a = np.load(events_fn)
    xs, ys, ts, ps = a['x'], a['y'], a['t'], a['p']

    t0 = T0_FRAME_ID
    t1 = T1_FRAME_ID

    # t=1 events are for 0->1
    # t=1001 events are for 1000->1001, which is same as 0->1, so it doesn't exist

    # ok so we want to transform 0 into 1: 1
    # then left is 1 (t0+1)
    # then right is 2 (t1+1)

    # 1 into 5: 2, 3, 4, 5
    # then left is 2 (t0+1)
    # then right is 6 (t1+1)

    # maybe a unit test could help with gt results
    # or sanity-test debugging t0=0, t1=1

    acc_sign = +1
    if t0 > t1:
        t0, t1 = t1, t0
        acc_sign = -1

    left = np.searchsorted(ts, t0+1)
    right = np.searchsorted(ts, t1+1)

    xs, ys, ts, ps = xs[left:right], ys[left:right], ts[left:right], ps[left:right]

    delta = np.zeros((H, W))
    accumulate(xs, ys, ts, ps, delta)
    delta = delta * THR

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
    # out = log_to_srgb((srgb_to_log(t0render_mono)-base-1e-6)//THR*THR+delta+base)
    # todo: double check
    # seems correct, since the ghost is fully banished when doing gt 0 to gt 1000
    # but slighly reduced contrast might be something
    if MODE == 'accurate':
        out = log_to_srgb(round_evts(srgb_to_log(t0render_mono)-base) + delta*acc_sign +base)
    elif MODE == 'direct':
        out = log_to_srgb(srgb_to_log(t0render_mono) + delta*acc_sign)
    else:
        assert False, f"unknown mode: {MODE}. expected 'accurate' or 'direct'"

    # stats(t0render_mono)
    # stats(t0render)
    # view_name = path.splitext(path.basename(events_fn))[0]
    # out_fn = path.join(OUT_DIR, view_name+f'_synth{T1_FRAME_ID:04d}.png')
    # Image.fromarray(t0render_mono.astype(np.uint8)).save(out_fn+'t0_mono.png')

    # stats(t0render_mono)
    # stats(srgb_to_log(t0render_mono))
    # stats(delta)
    # stats(srgb_to_log(t0render_mono)+delta)

    out = np.clip(out, 0, 255).astype(np.uint8)
    view_name = path.splitext(path.basename(t0render_fn))[0]
    # view = view_name
    # out_fn = path.join(OUT_DIR, f'r_{view}_{T1_FRAME_ID:04d}.png')
    if False and f'{T0_FRAME_ID:04d}' in view_name:
        out_fn = path.join(OUT_DIR, view_name.replace(f'{T0_FRAME_ID:04d}', f'{T1_FRAME_ID:04d}')+'.png')
    else:
        # out_fn = path.join(OUT_DIR, view_name+f'_synth{T1_FRAME_ID:04d}.png')
        out_fn = path.join(OUT_DIR, view_name+f'_from{T0_FRAME_ID:04d}_synth{T1_FRAME_ID:04d}.png')
    Image.fromarray(out).save(out_fn)
    out_fns.append(out_fn)

print()
print('Rendered following views:')
print(*out_fns, sep='\n')
