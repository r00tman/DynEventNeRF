#!/usr/bin/env bash

middle=(
cvprwfullrealablations_wolacclevent1_24-04-30_200_205_5fps_ls0.5_3e-2
cvprwfullrealablations_wolacclevent_24-04-30_200_205_5fps_ls0.5_3e-2
)

multiseg=(
)

for x in ${middle[@]}; do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_middle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done

# for x in ${multiseg[@]}; do
#     echo $x
#     # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_end --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
#     # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_middle_190_310 --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
#     # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
#     # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
#     python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --render_timestamp_frames 1000 --testskip 1 --config logs_auto/$x/args.txt
# done
