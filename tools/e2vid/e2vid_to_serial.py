#!/usr/bin/env python3
import os
import shutil
from tqdm import tqdm
import sys

inp_dir = sys.argv[1]
out_dir = sys.argv[1]+'_serial'
os.makedirs(out_dir, exist_ok=True)

idx = 0
for fn in tqdm(list(sorted(os.listdir(inp_dir)))):
    if not os.path.isfile(os.path.join(inp_dir, fn)) or os.path.splitext(fn)[-1] != '.png':
        print(os.path.splitext(fn)[-1] == '.png')
        continue
    outfn = '%05d.png'%idx

    shutil.copyfile(os.path.join(inp_dir, fn), os.path.join(out_dir, outfn))
    idx += 1


