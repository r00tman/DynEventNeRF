#!/usr/bin/env python3
import numpy as np
import os
from os import path
import sys
import shutil
from tqdm import tqdm

# inp = sys.argv[1]
# out = sys.argv[2]

# inp = './output/results_test_1000_?_serial'
# out = './data/dynsyn/spheres/train_e2vid/rgb'

# inp = './output/dvSaveExt-rec-2024_04_30_16_47_30_1714488450074116_80.0_90.0_?_serial'
# out = './data/dynsyn/24-04-30_80_90_5fps_ls0.5_3e-2/train_e2vid/rgb'

# inp = './output/dvSaveExt-rec-2024_04_30_16_47_30_1714488450074116_314.0_319.0_?_serial'
# out = './data/dynsyn/24-04-30_314_319_5fps_ls0.5_3e-2/train_e2vid/rgb'

inp = './output/dvSaveExt-rec-2024_04_30_17_02_49_1714489369704426_207.0_217.0_?_serial'
out = './data/dynsyn/24-04-30a_207_168_5fps_ls0.5_3e-2/train_e2vid/rgb'

# os.makedirs(out, exist_ok=True)

frames = [f'{f:04d}' for f in range(5, 1001, 5)]  # 10s sequence @ 20fps
# frames = [f'{f:04d}' for f in range(10, 1001, 10)]  # 5s sequence @ 20fps

for view in ['0', '1', '2', '3', '4', '5']:
    inpdir = inp.replace('?', view)
    files = os.listdir(inpdir)
    files.sort()
    # print(files)

    for frame, f in tqdm(list(zip(frames, files))):
        inpfn = path.join(inpdir, f)
        outfn = path.join(out, frame, f'r_{view}_{frame}.png')
        os.makedirs(path.join(out, frame), exist_ok=True)
        shutil.copyfile(inpfn, outfn)
