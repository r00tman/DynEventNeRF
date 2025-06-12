#!/usr/bin/env bash

set -euo pipefail
# set -x

common="--N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 90 --tend 210 --neg_ratio 0.9 --tonemap_eps 3e-2 --use_viewdirs False"

base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
# normal="--lambda_reg 1e-2 --damping_strength 0.93"
# e2vid="--is_rgb_only True --train_split train_5view_e2vid_nobg --lambda_reg 1e-2"
# lighter lambda_reg as 1e-2 resulted in transparent reconstructions, lower fps->lower influence of non-regularizer terms->...
# blurry="--is_rgb_only True --train_split train_5view_blurry --lambda_reg 1e-3"
edi="--is_rgb_only True --train_split train_5view --lambda_reg 1e-3"

for scene in 24-04-30_80_90_5fps_ls0.5_3e-2 24-04-30_314_319_5fps_ls0.5_3e-2 24-04-30a_207_168_5fps_ls0.5_3e-2; do
    sceneargs=""

    case $scene in
        24-04-30_80_90_5fps_ls0.5_3e-2)
            sceneargs="--tstart 190 --tend 310"
            ;;
        24-04-30_314_319_5fps_ls0.5_3e-2)
            sceneargs="--tstart 790 --tend 1000"
            ;;
        24-04-30a_207_168_5fps_ls0.5_3e-2)
            sceneargs="--tstart 190 --tend 310"
            ;;
        *)
            echo "No specific arguments for scene: $scene"
            ;;
    esac

    # ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealcomp_full_${scene} --scene data/dynsyn/${scene} $common $sceneargs $base $normal
    # ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealcomp_e2vid_${scene} --scene data/dynsyn/${scene} $common $sceneargs $base $e2vid
    # ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfullrealcomp_blurry_${scene} --scene data/dynsyn/${scene} $common $sceneargs $base $blurry
    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprwfullrealcomp_edi_${scene} --scene data/dynsyn/${scene} $common $sceneargs $base $edi
done
