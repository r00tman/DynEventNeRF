#!/usr/bin/env bash

middle=(
cvprshorts_24-04-29_100_105_5fps_ls0.5_3e-2
cvprshorts_24-04-29_114_124_5fps_ls0.5_3e-2
cvprshorts_24-04-30_200_205_5fps_ls0.5_3e-2
cvprshorts_24-04-30_256_266_5fps_ls0.5_3e-2
cvprshorts_24-04-30_332_342_5fps_ls0.5_3e-2
cvprshorts_24-04-30_80_90_5fps_ls0.5_3e-2
cvprshorts_24-04-30_858_868_5fps_ls0.5_3e-2
cvprshorts_24-04-30a_207_168_5fps_ls0.5_3e-2
)

end=(
cvprshorts_24-04-30_314_319_5fps_ls0.5_3e-2
cvprshorts_24-04-30_682_687_5fps_ls0.5_3e-2
)

for x in ${middle[@]}; do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_middle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done

for x in ${end[@]}; do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_end --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done
