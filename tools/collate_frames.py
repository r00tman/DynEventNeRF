#!/usr/bin/env python3
import numpy as np
import os
from os import path
import sys
import shutil
from tqdm import tqdm

inp = sys.argv[1]
out = sys.argv[2]

# inp = './data/dynsyn/spheres/train_blurry/rgb_direct'
# out = './data/dynsyn/spheres/train_blurry/rgb_new'

# os.makedirs(out, exist_ok=True)

files = os.listdir(inp)
files.sort()

for f in tqdm(files):
    _, cam, frame = path.splitext(f)[0].split('_')

    inpfn = path.join(inp, f)
    outfn = path.join(out, frame, f)
    os.makedirs(path.join(out, frame), exist_ok=True)
    shutil.copyfile(inpfn, outfn)
