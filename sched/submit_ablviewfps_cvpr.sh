#!/usr/bin/env bash

set -euo pipefail
# set -x

common="--N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5  --neg_ratio 0.9 --tonemap_eps 3e-2 --use_viewdirs False"

base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
fullmodel=${base}" --lambda_reg 1e-2 --damping_strength 0.93"

scene="24-04-30_200_205_5fps_ls0.5_3e-2"
splits=(
cvprabl_fps0.5
cvprabl_fps1
cvprabl_fps10
cvprabl_fps100
cvprabl_fps2
cvprabl_fps20
cvprabl_fps30
cvprabl_fps50
cvprabl_views2
cvprabl_views3
cvprabl_views4
)

for split in ${splits[@]}; do
# for scene in 24-04-30_80_90_5fps_ls0.5_3e-2 24-04-30_314_319_5fps_ls0.5_3e-2 24-04-30a_207_168_5fps_ls0.5_3e-2; do
    sceneargs=""
    
    case $scene in
        24-04-30_200_205_5fps_ls0.5_3e-2)
            sceneargs="--tstart 190 --tend 310"
            ;;
        24-04-29_100_105_5fps_ls0.5_3e-2)
            sceneargs="--tstart 90 --tend 210"
            ;;
        24-04-30_682_687_5fps_ls0.5_3e-2)
            sceneargs="--tstart 790 --tend 1000"
            ;;
        24-04-30_314_319_5fps_ls0.5_3e-2)
            sceneargs="--tstart 790 --tend 1000"
            ;;
        24-04-30_256_266_5fps_ls0.5_3e-2)
            sceneargs="--tstart 90 --tend 210"
            ;;
        24-04-30_332_342_5fps_ls0.5_3e-2)
            sceneargs="--tstart 190 --tend 310"
            ;;
        24-04-30_858_868_5fps_ls0.5_3e-2)
            sceneargs="--tstart 390 --tend 510"
            ;;
        24-04-30a_207_168_5fps_ls0.5_3e-2)
            sceneargs="--tstart 190 --tend 310"
            ;;
        24-04-30_80_90_5fps_ls0.5_3e-2)
            sceneargs="--tstart 190 --tend 310"
            ;;
        24-04-29_114_124_5fps_ls0.5_3e-2)
            sceneargs="--tstart 490 --tend 610"
            ;;
        *)
            echo "No arguments for scene: $scene"
            ;;
    esac

    ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvprfpsviews_${split}_${scene} --scene data/dynsyn/${scene} $common $sceneargs $fullmodel --train_split $split 

done
