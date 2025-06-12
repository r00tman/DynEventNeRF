#!/usr/bin/env bash

middle=(
cvprwfullrealcomp_edi_24-04-30_314_319_5fps_ls0.5_3e-2
cvprwfullrealcomp_edi_24-04-30_80_90_5fps_ls0.5_3e-2
cvprwfullrealcomp_edi_24-04-30a_207_168_5fps_ls0.5_3e-2
)

for x in ${middle[@]}; do
    echo $x
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_middle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done
