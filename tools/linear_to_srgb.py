#!/usr/bin/env python3
import os
from os import path
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

for fn in os.listdir(inp):
    a = np.array(Image.open(path.join(inp, fn)))

    result = linear_to_srgb(a)

    os.makedirs(out, exist_ok=True)
    Image.fromarray(result).save(path.join(out, fn))
