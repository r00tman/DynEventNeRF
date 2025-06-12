#!/usr/bin/env bash

real=(
    # ablations_hexplane_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_ngp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_no_accumulationloss_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_no_clipping_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_no_damping_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_no_eventloss_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_no_rgbloss_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_no_sparsity_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_trfcp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # # ablations_no_multiseg_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t0-1000_teps3e-2
    #
    #ablations_fps0.5_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_fps1_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_fps10_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_fps20_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_fps30_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_fps50_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_fps100_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_views2_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_views3_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_views4_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #ablations_views5_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #
    # ablations_renew_fps0.5_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_fps100_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_fps10_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_fps1_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_fps20_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_fps2_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_fps30_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_fps50_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_views2_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_views3_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_views4_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # ablations_renew_views5_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    # mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
    #
    # ablations_noacc_noeventloss_mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t90-210_teps3e-2
    #
    
    fullrealablations_full_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_hp_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_ngp_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_trfcp_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_woclipping_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_wodamping_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_wolacc_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_wolevent_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_wolrgb_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_wolrgblacc_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_wolrgblevent_24-04-30_200_205_5fps_ls0.5_3e-2
    fullrealablations_wolsparsity_24-04-30_200_205_5fps_ls0.5_3e-2
)

synth=(
    # ablations_hexplane_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_ngp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_no_accumulationloss_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_no_clipping_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_no_eventloss_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_no_rgbloss_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_no_sparsity_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_trfcp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    #
    # ablations_fps1_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps2_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps5_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps10_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps20_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps30_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps50_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps100_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    #
    # ablations_fps100_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps10_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps1_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps20_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps2_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps30_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps50_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # ablations_fps5_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    # 
    # ablations_noacc_noeventloss_mlp_dress_rounded_lambdaanneal30k_1e-2_nolrsched_teps1e-2_cropr0.4
    #
    # fullablations_full_blender
    # fullablations_full_dress
    # fullablations_full_lego_dyn2
    # fullablations_full_lego_dyn2_static
    # fullablations_full_spheres
    # fullablations_hp_blender
    # fullablations_hp_dress
    # fullablations_hp_lego_dyn2
    # fullablations_hp_lego_dyn2_static
    # fullablations_hp_spheres
    # fullablations_ngp_blender
    # fullablations_ngp_dress
    # fullablations_ngp_lego_dyn2
    # fullablations_ngp_lego_dyn2_static
    # fullablations_ngp_spheres
    # fullablations_trfcp_blender
    # fullablations_trfcp_dress
    # fullablations_trfcp_lego_dyn2
    # fullablations_trfcp_lego_dyn2_static
    # fullablations_trfcp_spheres
    # fullablations_woclipping_blender
    # fullablations_woclipping_dress
    # fullablations_woclipping_lego_dyn2
    # fullablations_woclipping_lego_dyn2_static
    # fullablations_woclipping_spheres
    # fullablations_wolacc_blender
    # fullablations_wolacc_dress
    # fullablations_wolacc_lego_dyn2
    # fullablations_wolacc_lego_dyn2_static
    # fullablations_wolacc_spheres
    # fullablations_wolevent_blender
    # fullablations_wolevent_dress
    # fullablations_wolevent_lego_dyn2
    # fullablations_wolevent_lego_dyn2_static
    # fullablations_wolevent_spheres
    # fullablations_wolrgb_blender
    # fullablations_wolrgb_dress
    # fullablations_wolrgblacc_blender
    # fullablations_wolrgblacc_dress
    # fullablations_wolrgblacc_lego_dyn2
    # fullablations_wolrgblacc_lego_dyn2_static
    # fullablations_wolrgblacc_spheres
    # fullablations_wolrgb_lego_dyn2
    # fullablations_wolrgb_lego_dyn2_static
    # fullablations_wolrgblevent_blender
    # fullablations_wolrgblevent_dress
    # fullablations_wolrgblevent_lego_dyn2
    # fullablations_wolrgblevent_lego_dyn2_static
    # fullablations_wolrgblevent_spheres
    # fullablations_wolrgb_spheres
    # fullablations_wolsparsity_blender
    # fullablations_wolsparsity_dress
    # fullablations_wolsparsity_lego_dyn2
    # fullablations_wolsparsity_lego_dyn2_static
    # fullablations_wolsparsity_spheres

    # fullablations_full_blender
    # fullablations_full_lego_dyn2
    # fullablations_full_lego_dyn2_static
    # fullablations_full_spheres
    # fullablations_hp_lego_dyn2
    # fullablations_ngp_lego_dyn2
    # fullablations_trfcp_lego_dyn2
    # fullablations_trfcp_spheres
    # fullablations_woclipping_lego_dyn2
    # fullablations_woclipping_spheres
    # fullablations_wolacc_lego_dyn2
    # fullablations_wolevent_spheres
    # fullablations_wolrgblacc_blender
    # fullablations_wolrgblacc_dress
    # fullablations_wolrgblacc_lego_dyn2
    # fullablations_wolrgblacc_spheres
    # fullablations_wolrgblevent_blender
    # fullablations_wolrgblevent_spheres
    # fullablations_wolsparsity_dress
    # fullablations_wolsparsity_lego_dyn2
    # fullablations_wolsparsity_lego_dyn2_static
    # fullablations_wolsparsity_spheres
)

for x in ${synth[@]}; do
# for x in $(ls fullablations_rerender); do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --render_timestamp_frames 20 --config logs_auto/$x/args.txt
done

for x in ${real[@]}; do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_middle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done
