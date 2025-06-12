#!/usr/bin/env python3
import os
from os import path
import sys
import glob
import shutil
from tqdm import tqdm

import numpy as np

outdir = sys.argv[1]
views = sys.argv[2:]
views.sort()

print('out dir:', outdir)
print('views:', views)

with open('frame_lists/33_frames.txt') as f:
    frame_list = [int(l.strip()) for l in f.readlines() if l.strip()]


os.makedirs(outdir, exist_ok=True)

for view_idx, view in enumerate(views):
    print(f'doing view {view_idx}: {view}')
    files = glob.glob(path.join(view, '*.png'))
    files.sort()
    print(files)

    # frame 0 does not exist, use frame 1 as a substitute
    files = [files[0]] + files

    assert len(files) == len(frame_list)
    for frame_idx, fn in zip(frame_list, tqdm(files)):
        shutil.copyfile(fn, path.join(outdir, f'r_{view_idx:02d}_{frame_idx:04d}.png'))
