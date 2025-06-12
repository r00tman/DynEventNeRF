#!/usr/bin/env bash

set -euo pipefail
# set -x

common="--train_split train_rounded --N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 0 --tend 1000 --neg_ratio 0.9 --tonemap_eps 1e-2 --use_viewdirs False --damping_strength 1.0"

base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
fullmodel=${base}" --lambda_reg 1e-2"

trfcp="--config configs/tensorf5_lambda1e-3_progressive_500_nosparsity.txt --lambda_reg 1e-2"
ngp="--config configs/ngp.txt --lambda_reg 1e-2"
hexplane="--config configs/tensorfvm3.txt --lambda_reg 1e-2 --lrate 1e-2"

wolevent=${fullmodel}" --use_event_loss False"
wolrgb=${fullmodel}" --use_rgb_loss False"
wolacc=${fullmodel}" --use_accumulation_loss False"
wolsparsity=${base}" --lambda_reg 0"
woclipping=${fullmodel}" --crop_r 1.0 --crop_y_max 1.0 --crop_y_min -1.0"
wolrgblevent=${fullmodel}" --use_rgb_loss False --use_event_loss False" 
wolrgblacc=${fullmodel}" --use_rgb_loss False --use_accumulation_loss False"
wolacclevent=${fullmodel}" --use_accumlation_loss False --use_event_loss False"

for scene in dress spheres blender lego_dyn2 lego_dyn2_static; do
    sceneargs=
    if [[ ${scene} == blender ]]; then
        sceneargs="--bg_color 70 --crop_r 0.2"
    elif [[ ${scene} == dress ]]; then
        sceneargs="--crop_r 0.4"
    fi

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_full_${scene} --scene data/dynsyn/${scene} $common $sceneargs $fullmodel

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_trfcp_${scene} --scene data/dynsyn/${scene} $common $sceneargs $trfcp
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_ngp_${scene} --scene data/dynsyn/${scene} $common $sceneargs $ngp
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_hp_${scene} --scene data/dynsyn/${scene} $common $sceneargs $hexplane

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_wolevent_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolevent
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_wolrgb_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolrgb
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_wolacc_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolacc
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_wolsparsity_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolsparsity
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_woclipping_${scene} --scene data/dynsyn/${scene} $common $sceneargs $woclipping
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_wolrgblevent_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolrgblevent
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_wolrgblacc_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolrgblacc
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname fullablations_wolacclevent_${scene} --scene data/dynsyn/${scene} $common $sceneargs $wolacclevent

done
