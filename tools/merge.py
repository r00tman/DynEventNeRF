#!/usr/bin/env python3
import numpy as np
from PIL import Image
import os
import sys
from glob import glob
from os import path
import re

# here we have r_{view}_*.png
# we need to group them by {view}

def get_view_by_fn(x):
    return re.match(r'r_([0-9]*).*', path.basename(x)).group(1)

# print(get_view_by_fn('r_01_43243.png'))
# exit()

inpdir = sys.argv[1]
outdir = sys.argv[2]
# inpdir = "/CT/EventNeRF/work/dynamic/code/data/dynsyn/lego_dyn2/train_frame_500_alt/prergb"
# outdir = "/CT/EventNeRF/work/dynamic/code/data/dynsyn/lego_dyn2/train_frame_500_alt/rgb"

views = dict()
for f in glob(path.join(inpdir, '*')):
    views.setdefault(get_view_by_fn(f), []).append(f)

print(views)

bg_color = 159.

for view, refs in views.items():
    outfn = path.join(outdir, f'r_{view}_0000.png')

    a = np.array(Image.open(refs[0]))
    adist = abs(a-bg_color)


    for bfn in refs[1:]:
        b = np.array(Image.open(bfn))
        bdist = abs(b-bg_color)

        amask = adist<=bdist
        bmask = ~amask

        a = a*amask+b*bmask
        adist = abs(a-bg_color)

    Image.fromarray(a).save(outfn)

