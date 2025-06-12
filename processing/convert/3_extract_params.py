#!/usr/bin/env python3
import numpy as np
import sys
import os
from os import path

W = 346

inpfile = sys.argv[1]
intrpath = sys.argv[2]
posepath = sys.argv[3]

with open(inpfile) as f:
    currentCamera = None
    remainingIntLines = 0
    remainingExtLines = 0
    for line in f:
        if remainingIntLines > 0:
            vals = [float(x) for x in line.strip().split()[1:]]
            intrinsics.append(vals)
            del vals
            remainingIntLines -= 1
            if remainingIntLines == 0:
                tmp = np.array(intrinsics)
                tmp[:2,:] *= W  # keep the third line 0 0 1
                intrinsics = np.eye(5)
                intrinsics[:3, :3] = tmp
                intrinsics[4] = distortion
                print('int', currentCamera)
                print(intrinsics)

                outDir = intrpath
                os.makedirs(outDir, exist_ok=True)
                np.savetxt(path.join(outDir, f'{currentCamera}.txt'), intrinsics)
                del tmp, intrinsics

        elif remainingExtLines > 0:
            vals = [float(x) for x in line.strip().split()[1:]]
            extrinsics.append(vals)
            del vals
            remainingExtLines -= 1
            if remainingExtLines == 0:
                extrinsics = np.array(extrinsics)
                print('ext', currentCamera)
                print(extrinsics)

                outDir = posepath
                os.makedirs(outDir, exist_ok=True)
                np.savetxt(path.join(outDir, f'{currentCamera}.txt'), extrinsics)
                del extrinsics

        else:
            command = line.strip().split()[:1]
            command = command[0] if len(command)>0 else ''
            if command == 'camera':
                _, camIdx, camName = line.strip().split()
                currentCamera = f'{camName}_idx{camIdx}'
                distortion = None
                intrinsics = []
                extrinsics = []
                del camIdx, camName
            elif 'intrinsics' in line:
                remainingIntLines = 3
            elif 'extrinsics' in line:
                remainingExtLines = 3
            elif command == 'distortion':
                _, k1, k2, p1, p2, k3 = line.strip().split()
                distortion = np.array([float(k1), float(k2), float(p1), float(p2), float(k3)])
                del _, k1, p1, p2, k3
            del command
