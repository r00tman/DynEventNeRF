#!/usr/bin/env python3
import os
from os import path
from glob import glob
import sys
import numpy as np
from PIL import Image
import cv2

inp, out = sys.argv[1:]


def linear_to_srgb(linear):
    '''
    linear: 0-255 [H, W]
    turns log brightness into 0-1 abs then to 0-255 srgb
    '''
    linear = linear / 255.
    a = np.maximum(0, linear) ** (1 / 2.2)
    a = np.clip(a*255, 0, 255).astype(np.uint8)
    return a

for (root, dirs, files) in os.walk(inp):
    rel_root = path.relpath(root, inp)
    for file in files:
        fn = path.join(root, file)
        a = np.array(Image.open(fn))
        if a.shape[-1] == 3:
            a = a[..., 0]

        default = cv2.cvtColor(a, cv2.COLOR_BayerBG2RGB)
        vng = cv2.cvtColor(a, cv2.COLOR_BayerBG2RGB_VNG)

        # correct vng y=-2 shift
        result = np.copy(default)
        result[2:] = vng[:-2]

        result = linear_to_srgb(result)

        out_dir = path.join(out, rel_root)
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(result).save(path.join(out_dir, file))
