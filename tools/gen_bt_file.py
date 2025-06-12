#!/usr/bin/env python3
import numpy as np
# with open('frame_lists/33_frames.txt') as f:
#     res1 = np.array([int(c.strip()) for c in f], dtype=np.int64)

with open('frame_lists_100/33_frames.txt') as f:
    res1 = np.array([int(c.strip()) for c in f], dtype=np.int64)[::-1]

# with open('frame_lists/65_frames.txt') as f:
#     res2 = np.array([int(c.strip()) for c in f], dtype=np.int64)

# with open('frame_lists/65_frames.txt') as f:
#     res1 = np.array([int(c.strip()) for c in f], dtype=np.int64)

with open('frame_lists/129_frames.txt') as f:
    res2 = np.array([int(c.strip()) for c in f], dtype=np.int64)

# with open('frame_lists/257_frames.txt') as f:
#     res3 = np.array([int(c.strip()) for c in f], dtype=np.int64)

# res1 = np.tile(res1[..., None], (1, 2)).ravel()[:-1]
res1 = np.tile(res1[..., None], (1, 4)).ravel()[:-1]
# res1 = res2
# res3 = np.arange(0, 1001)
# res1, res2 = res3, res3
# res2 = res1

for a, b in zip(res1, res2):
    # print(f'logs_auto/auto_slurm_3kits_lowertrfiters/train_{a:05d}_auto/args.txt:train_:r_{b:05d}.png')
    # print(f'logs_auto/lego_noprefilter/train_{a:05d}_auto/args.txt:train_:r_{b:05d}.png')
    # print(f'logs_auto/sph_noprefilter_notrans1/train_{a:05d}_auto/args.txt:train_:r_{b:05d}.png')
    # print(f'logs_auto/real1_mlp2_lambda1e-3_/train_{a:05d}_auto/args.txt:circle:r_{b:05d}.png')
    print(f'logs_auto/real1_mlp2_lambda1e-3_goodundist/train_{a:05d}_auto/args.txt:circle:r_{b:05d}.png')
