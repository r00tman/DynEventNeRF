#!/usr/bin/env bash

set -euo pipefail
# set -x

common="--train_split train_5view --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 90 --tend 210 --neg_ratio 0.9 --tonemap_eps 3e-2 --use_viewdirs False"
common_womultiseg="--train_split train_5view --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 0 --tend 1000 --neg_ratio 0.9 --tonemap_eps 3e-2 --use_viewdirs False"

base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
fullmodel=${base}" --lambda_reg 1e-2 --damping_strength 0.93"

trfcp="--config configs/tensorf5_lambda1e-3_progressive_500_nosparsity.txt --lambda_reg 1e-2 --damping_strength 0.93"
ngp="--config configs/ngp.txt --lambda_reg 1e-2 --damping_strength 0.93"
hexplane="--config configs/tensorfvm3.txt --lambda_reg 1e-2 --lrate 1e-2 --damping_strength 0.93"

wolevent=${fullmodel}" --use_event_loss False"
wolrgb=${fullmodel}" --use_rgb_loss False"
wolacc=${fullmodel}" --use_accumulation_loss False"
wolsparsity=${base}" --lambda_reg 0 --damping_strength 0.93"
woclipping=${fullmodel}" --crop_r 1.0 --crop_y_max 1.0 --crop_y_min -1.0"
wolrgblevent=${fullmodel}" --use_rgb_loss False --use_event_loss False" 
wolrgblacc=${fullmodel}" --use_rgb_loss False --use_accumulation_loss False"
wolacclevent=${fullmodel}" --use_accumlation_loss False --use_event_loss False"
wodamping=${base}" --lambda_reg 1e-2 --damping_strength 1.0"

for scene in 24-04-30_200_205_5fps_ls0.5_3e-2; do
# for scene in 24-04-30_80_90_5fps_ls0.5_3e-2 24-04-30_314_319_5fps_ls0.5_3e-2 24-04-30a_207_168_5fps_ls0.5_3e-2; do
    sceneargs=

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_full_${scene} --scene data/dynsyn/${scene} $common $sceneargs $fullmodel

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_trfcp_${scene} --scene data/dynsyn/${scene} $common $sceneargs $trfcp
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_ngp_${scene} --scene data/dynsyn/${scene} $common $sceneargs $ngp
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_hp_${scene} --scene data/dynsyn/${scene} $common $sceneargs $hexplane

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wolevent_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolevent
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wolrgb_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolrgb
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wolacc_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolacc
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wolsparsity_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolsparsity
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_woclipping_${scene} --scene data/dynsyn/${scene} $common $sceneargs $woclipping
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wolrgblevent_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolrgblevent
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wolrgblacc_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolrgblacc
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wolacclevent_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolacclevent
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_wodamping_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wodamping

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealablations_womultiseg_${scene} --scene data/dynsyn/${scene} $common_womultiseg $sceneargs $fullmodel
done
