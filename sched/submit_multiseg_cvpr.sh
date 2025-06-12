#!/usr/bin/env bash

set -euo pipefail
# set -x

common="--train_split train_5view --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5  --neg_ratio 0.9 --tonemap_eps 3e-2 --use_viewdirs False"

base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
fullmodel=${base}" --lambda_reg 1e-2 --damping_strength 0.93"

scenes=(
#24-04-30_200_205_5fps_ls0.5_3e-2  
24-04-30a_207_168_5fps_ls0.5_3e-2 
)

for scene in ${scenes[@]}; do
# for scene in 24-04-30_80_90_5fps_ls0.5_3e-2 24-04-30_314_319_5fps_ls0.5_3e-2 24-04-30a_207_168_5fps_ls0.5_3e-2; do
    sceneargs=""
    
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprmultiseg_${scene}_0_210 --scene data/dynsyn/${scene} $common $sceneargs $fullmodel --tstart 0 --tend 210
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprmultiseg_${scene}_190_410 --scene data/dynsyn/${scene} $common $sceneargs $fullmodel --tstart 190 --tend 410
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprmultiseg_${scene}_390_610 --scene data/dynsyn/${scene} $common $sceneargs $fullmodel --tstart 390 --tend 610
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprmultiseg_${scene}_590_810 --scene data/dynsyn/${scene} $common $sceneargs $fullmodel --tstart 590 --tend 810
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprmultiseg_${scene}_790_1000 --scene data/dynsyn/${scene} $common $sceneargs $fullmodel --tstart 790 --tend 1000

done
