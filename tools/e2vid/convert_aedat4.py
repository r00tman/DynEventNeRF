#!/usr/bin/env python3
import numpy as np
from dv import AedatFile
from tqdm import tqdm, trange
import cv2
from zipfile import ZipFile
import io

with AedatFile('./toym-2022_08_02_20_00_47.aedat4') as f:
    events = np.hstack([packet for packet in f['events_1'].numpy()])

# mask = cv2.imread('./mask.png')
# if len(mask.shape) == 3:
#     mask = mask[..., 0]

print('loaded')
zipfn = 'swag.zip'
txtfn = 'swag.txt'

with ZipFile(zipfn, 'w') as myzip:
    out = io.StringIO()
    # with open('drums.txt', 'w') as out:
    if True:
        print(346, 260, file=out)

        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']


        print('saving...')
        for i in trange(len(timestamps)):
            # if timestamps[i] > 20*1000000 + timestamps[0]:
            #     print(i)
            #     break
            # if mask[y[i], x[i]] > 0:
                print('%.12f'%(timestamps[i]/1000000.), x[i], y[i], polarities[i], file=out)


        # np.savetxt(out, data)

    print('zipping...')
    myzip.writestr(txtfn, out.getvalue())
