#!/usr/bin/env bash

middle=(
cvprfpsviews_cvprabl_views2_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_views3_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_views4_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps0.5_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps1_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps2_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps10_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps20_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps30_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps50_24-04-30_200_205_5fps_ls0.5_3e-2
cvprfpsviews_cvprabl_fps100_24-04-30_200_205_5fps_ls0.5_3e-2
)

for x in ${middle[@]}; do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_middle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done
