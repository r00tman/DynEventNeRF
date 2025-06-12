#!/usr/bin/env bash
common="--N_iters 150001 --N_anneal_lambda 30000 --use_lr_scheduler False --event_threshold 0.5 --tstart 0 --tend 1000 --neg_ratio 0.9 --tonemap_eps 1e-2 --use_viewdirs False --damping_strength 1.0"

base="--config configs/mlp2_lambda1e-3.txt --lrate 1e-4 --max_freq_log2_pos 14 --max_freq_log2_time 7"
fullmodel=${base}" --lambda_reg 1e-2"

#scenes=dress spheres blender lego_dyn2 lego_dyn2_static
# scenes=dress
# scenes=blender
scenes=(dress blender)

for scene in ${scenes[@]}; do
    # for res in 4xres; do
    for res in 4xres 16xres; do
        sceneargs=
        if [[ ${scene} == blender ]]; then
            sceneargs="--bg_color 70 --crop_r 0.2"
        elif [[ ${scene} == dress ]]; then
            sceneargs="--crop_r 0.4"
        fi
            
        # python ./ddp_train_nerf.py --expname cvpr25r_abl_synth_res_${scene}_${res} --scene data/dynsyn/${scene} --train_split train_${res}_rounded $common $sceneargs $fullmodel
        # ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvpr25r_abl_synth_res_${scene}_${res} --scene data/dynsyn/${scene} --train_split train_${res}_rounded $common $sceneargs $fullmodel
        ./scripts/launcher_multi_archive.sh python ./ddp_train_nerf.py --expname cvpr25r_abl_synth_res_${scene}_${res} --scene data/dynsyn/${scene} --train_split train_${res}_rounded $common $sceneargs $fullmodel --i_weights 500
    done
done
