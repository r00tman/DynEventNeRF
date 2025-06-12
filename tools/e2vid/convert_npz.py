#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm, trange
import cv2
from zipfile import ZipFile
import io
import sys
from os import path


infn = sys.argv[1]
ts_per_second = float(sys.argv[2]) if len(sys.argv) > 2 else 1000
t_min = float(sys.argv[3]) if len(sys.argv) > 3 else -1
t_max = float(sys.argv[4]) if len(sys.argv) > 4 else -1
print('loading', infn, 'ts_per_second', ts_per_second, 't_min', t_min, 't_max', t_max)
events = np.load(infn)
xs, ys, ts, ps = events['x'], events['y'], events['t'], events['p']
# mask = cv2.imread('./mask.png')
# if len(mask.shape) == 3:
#     mask = mask[..., 0]

print('loaded')
zipfn = path.splitext(path.basename(infn))[0]+'.zip'
txtfn = path.splitext(path.basename(infn))[0]+'.txt'

with ZipFile(zipfn, 'w') as myzip:
    out = io.StringIO()
    # with open('drums.txt', 'w') as out:
    if True:
        print(346, 260, file=out)

        # timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        x, y, polarities = xs, ys, ps
        timestamps = ts


        print('saving...')
        for i in trange(len(timestamps)):
            # if timestamps[i] > 20*1000000 + timestamps[0]:
            #     print(i)
            #     break
            # if mask[y[i], x[i]] > 0:
                # print('%.12f'%timestamps[i], x[i], y[i], polarities[i], file=out)
                # print('%.12f'%(timestamps[i]/1000), x[i], y[i], polarities[i], file=out)
                # if t_min > 0 and timestamps[i] < t_min:
                #     continue
                # if t_max > 0 and timestamps[i] > t_max:
                #     continue
                if t_min > 0 and t_max > 0:
                    ts = (ts-t_min)/(t_max-t_min)
                    ts = timestamps[i]/ts_per_second
                    if ts < 0 or ts > 1:
                        continue
                else:
                    ts = timestamps[i]/ts_per_second
                # print('%.12f'%(timestamps[i]/ts_per_second), x[i], y[i], polarities[i], file=out)
                print('%.12f'%(ts), x[i], y[i], polarities[i], file=out)


        # np.savetxt(out, data)

    print('zipping...')
    myzip.writestr(txtfn, out.getvalue())
