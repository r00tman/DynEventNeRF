#!/usr/bin/env bash

synth=(

cvpr25r_abl_synth_res_blender_16xres
cvpr25r_abl_synth_res_blender_4xres
cvpr25r_abl_synth_res_dress_16xres
cvpr25r_abl_synth_res_dress_4xres
# cvpr25r_abl_synth_views1_blender_12
# cvpr25r_abl_synth_views1_blender_2
cvpr25r_abl_synth_views1_blender_24
# cvpr25r_abl_synth_views1_blender_3
cvpr25r_abl_synth_views1_blender_36
# cvpr25r_abl_synth_views1_blender_4
cvpr25r_abl_synth_views1_blender_48
# cvpr25r_abl_synth_views1_blender_6
# cvpr25r_abl_synth_views1_dress_12
# cvpr25r_abl_synth_views1_dress_2
cvpr25r_abl_synth_views1_dress_24
# cvpr25r_abl_synth_views1_dress_3
cvpr25r_abl_synth_views1_dress_36
# cvpr25r_abl_synth_views1_dress_4
cvpr25r_abl_synth_views1_dress_48
# cvpr25r_abl_synth_views1_dress_6

    # cvpr25r_abl_synth_views1_dress_12
    # cvpr25r_abl_synth_views1_dress_2
    # cvpr25r_abl_synth_views1_dress_24
    # cvpr25r_abl_synth_views1_dress_3
    # cvpr25r_abl_synth_views1_dress_36
    # cvpr25r_abl_synth_views1_dress_4
    # cvpr25r_abl_synth_views1_dress_48
    # cvpr25r_abl_synth_views1_dress_6

    # cvpr25r_abl_synth_views1_blender_12
    # cvpr25r_abl_synth_views1_blender_2
    # cvpr25r_abl_synth_views1_blender_24
    # cvpr25r_abl_synth_views1_blender_3
    # cvpr25r_abl_synth_views1_blender_36
    # cvpr25r_abl_synth_views1_blender_4
    # cvpr25r_abl_synth_views1_blender_48
    # cvpr25r_abl_synth_views1_blender_6

    # cvpr25r_abl_synth_views1_spheres_12
    # cvpr25r_abl_synth_views1_spheres_2
    # cvpr25r_abl_synth_views1_spheres_24
    # cvpr25r_abl_synth_views1_spheres_3
    # cvpr25r_abl_synth_views1_spheres_36
    # cvpr25r_abl_synth_views1_spheres_4
    # cvpr25r_abl_synth_views1_spheres_48
    # cvpr25r_abl_synth_views1_spheres_6
)

for x in ${synth[@]}; do
# for x in $(ls fullablations_rerender); do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --render_timestamp_frames 20 --config logs_auto/$x/args.txt
done
