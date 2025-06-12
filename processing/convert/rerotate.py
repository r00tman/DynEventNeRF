#!/usr/bin/env python3
import numpy as np
import json
import cv2
import os
from os import path

jsondict = json.load(open('./camdict.json'))
names = list(sorted(jsondict.keys()))
w2c = [np.linalg.inv(np.array(jsondict[n]['W2C']).reshape(4,4)) for n in names]

print(w2c)
center, a, b = w2c[3][:3,3], w2c[4][:3,3], w2c[1][:3,3]

up = np.cross(a-center, b-center)
up = up/np.linalg.norm(up)
print(up)

targetup = np.array([0, 1, 0])

ang = np.arccos(np.dot(targetup, up))
axis = np.cross(up, targetup)
axis = axis / np.linalg.norm(axis)

rot = np.array(cv2.Rodrigues(axis*ang)[0])

print(rot@up)
print(w2c[0][:3].shape, rot.shape)
# approach 1: correct new up, bad directions
neww2c = [np.r_[rot@mat[:3], mat[3:4]] for mat in w2c]
# approach 2
# rot = np.r_[np.c_[rot, np.array([0,0,0])], np.array([[0,0,0,1]])]
# neww2c = [mat@rot for mat in w2c]
# approach 3
# neww2c = [np.linalg.inv(np.r_[np.linalg.inv(rot)@np.linalg.inv(mat)[:3], mat[3:4]]) for mat in w2c]
assert neww2c[0].shape == (4,4)

center, a, b = neww2c[3][:3,3], neww2c[4][:3,3], neww2c[1][:3,3]
up = np.cross(a-center, b-center)
up = up/np.linalg.norm(up)
print('new up', up)

for name, newmat in zip(names, neww2c):
    jsondict[name]['W2C'] = np.linalg.inv(newmat).flatten().tolist()
print(jsondict)
with open('camdict_rot.json', 'w') as f:
    json.dump(jsondict, f)

outdir = 'pose_norm'
os.makedirs(outdir, exist_ok=True)
for name, newmat in zip(names, neww2c):
    np.savetxt(path.join(outdir, path.splitext(name)[0]+'.txt'), np.linalg.inv(newmat))
