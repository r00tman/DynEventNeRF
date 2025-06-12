#!/usr/bin/env bash

start=(
cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_0_210
cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_0_210
)
middle=(
cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_190_410
cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_390_610
cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_590_810
cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_190_410
cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_390_610
cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_590_810
)

end=(
cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_790_1000
cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_790_1000
)

for x in ${start[@]}; do
    echo $x
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_start --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test_5view --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done

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
