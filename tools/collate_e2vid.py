#!/usr/bin/env python3
import numpy as np
import os
from os import path
import sys
import shutil
from tqdm import tqdm

# inp = sys.argv[1]
# out = sys.argv[2]

inp = './output/results_test_1000_?_serial'
out = './data/dynsyn/spheres/train_e2vid/rgb'

# os.makedirs(out, exist_ok=True)

frames = [f'{f:04d}' for f in range(50, 1001, 50)]

for view in ['00', '01', '02', '03', '04']:
    inpdir = inp.replace('?', view)
    files = os.listdir(inpdir)
    files.sort()
    # print(files)

    for frame, f in tqdm(list(zip(frames, files))):
        inpfn = path.join(inpdir, f)
        outfn = path.join(out, frame, f'r_{view}_{frame}.png')
        os.makedirs(path.join(out, frame), exist_ok=True)
        shutil.copyfile(inpfn, outfn)
