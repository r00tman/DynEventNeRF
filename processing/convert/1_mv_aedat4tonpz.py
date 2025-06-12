#!/usr/bin/env python3
import dvpstat_cpp
import numpy as np
import cv2
import dv_processing as dvp
import os
from os import path
import sys
from tqdm import tqdm
from dvputils import getStreamByCameraId

infn = sys.argv[1]
t0 = int(sys.argv[2])
start = float(sys.argv[3])
end = float(sys.argv[4])
out_path = sys.argv[5]

TIMESTAMP_PER_SEC = 1e6

start *= TIMESTAMP_PER_SEC
end *= TIMESTAMP_PER_SEC


def dumpEvents(mcr):
    res = []
    with tqdm(total=(end-start)/1e6) as pbar:
        lastupd = 0
        while mcr.isRunning():
            events = mcr.getNextEventBatch()


            if events is not None:
                newupd = events.getHighestTime()-t0-start
                pbar.update((newupd-lastupd)/1e6)
                lastupd = newupd

                if events.getHighestTime() < t0 + start:
                    continue
                if events.getLowestTime() > t0 + end:
                    break
                # print(f"{events}")
                res.append(events.numpy())

    res = np.hstack(res)
    return res

for camid in range(6):
    print(f'reading camera {camid}...')
    mcr = getStreamByCameraId(infn, camid)

    events = dumpEvents(mcr)
    events['timestamp'] -= t0
    events['polarity'] = events['polarity'] * 2 - 1

    # 1s = 1e6 timestamp
    print(events['timestamp'].max(), 'max timestamp')

    time_mask = ((events['timestamp'] >= start) & (events['timestamp'] < end))
    trimmed = events[time_mask]
    print(trimmed.size/events.size, 'trimmed size/events size')

    trimmed['timestamp'] = np.floor(trimmed['timestamp'].astype(np.float64)-start).astype(trimmed['timestamp'].dtype)
    print(trimmed['timestamp'].min(), trimmed['timestamp'].max(), 'timestamp min', 'timestamp max')

    final = trimmed

    xs = np.array(final['x'], dtype=np.int64)
    ys = np.array(final['y'], dtype=np.int64)
    ts = np.array(final['timestamp'], dtype=np.float64)/(end-start)*1000
    print(ts.min(), ts.max(), 'tsmin', 'tsmax')
    ps = np.array(final['polarity'], dtype=np.int64)

    # out_path = path.join(path.dirname(path.dirname(infn)), 'events')
    # out_path = path.join(path.dirname(infn), 'events')
    # out_path = path.splitext(path.basename(infn))[0]
    os.makedirs(out_path, exist_ok=True)

    np.savez(path.join(out_path, path.splitext(path.basename(infn))[0]+f'_{t0}_{start/1e6}_{end/1e6}_{camid}.npz'), x=xs, y=ys, t=ts, p=ps)

