#!/usr/bin/env bash

middle=(
mlp_24-04-29_59_69_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t290-410_teps3e-2
mlp_24-04-29_114_124_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t490-610_teps3e-2
mlp_24-04-29_165_170_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-410_teps3e-2
mlp_24-04-30_80_90_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2 
mlp_24-04-30_256_266_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t90-210_teps3e-2 
mlp_24-04-30_332_342_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
mlp_24-04-30_438_448_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t390-510_teps3e-2
mlp_24-04-30_567_577_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t90-210_teps3e-2
mlp_24-04-30a_117_127_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t90-210_teps3e-2
mlp_24-04-30a_163_168_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t290-510_teps3e-2
mlp_24-04-30a_207_168_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t190-310_teps3e-2
)

end=(
mlp_24-04-30_314_319_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t790-1000_teps3e-2
mlp_24-04-30_682_687_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t790-1000_teps3e-2
)

for x in ${middle[@]}; do
    echo $x
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_middle --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done

for x in ${end[@]}; do
    echo $x
    # ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split circle_end --write_video True --render_bullet_time True --testskip 10 --config logs_auto/$x/args.txt
    ./scripts/launcher.sh python ./ddp_test_nerf_video.py --render_split test --write_video True --render_bullet_time False --testskip 1 --config logs_auto/$x/args.txt
done
