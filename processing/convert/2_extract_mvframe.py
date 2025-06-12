#!/usr/bin/env python3
import numpy as np
import cv2
import dv
import os
from os import path
import sys
from tqdm import tqdm
from PIL import Image
from dvputils import getStreamByCameraId

infn = sys.argv[1]
t0 = int(sys.argv[2])
timestamp = float(sys.argv[3])
out_path = sys.argv[4]

# return frame f_i such that i=argmax_i t_i<=timestamp

TIMESTAMP_PER_SEC = 1e6

timestamp *= TIMESTAMP_PER_SEC

for camid in range(6):
    print(f'reading camera {camid}...')
    mcr = getStreamByCameraId(infn, camid)
    goodframe = None
    while mcr.isRunning():
        frame = mcr.getNextFrame()
        ts = frame.timestamp  # start of exposure
        # print(ts-t0, timestamp)
        if ts-t0 > timestamp:
            break
        goodframe = frame
    print('good')
    print(goodframe.timestamp-t0, goodframe.timestamp+goodframe.exposure.microseconds-t0, timestamp)

    print(goodframe.image)
    img = np.tile(goodframe.image[..., None], (1, 1, 3))

    os.makedirs(out_path, exist_ok=True)

    out_fn = path.join(out_path, path.splitext(path.basename(infn))[0]+f'_{camid}_{goodframe.timestamp-t0}.png')
    Image.fromarray(img).save(out_fn)

