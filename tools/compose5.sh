#!/usr/bin/env bash
ffmpeg -y -i "$1" -i "$2" -i "$3" -i "$4" -i "$5" -filter_complex "[0][1][2][3][4]hstack=inputs=5[vout]" -map "[vout]" -crf 10 test_comp_$(basename "$(pwd)").mp4
# ffmpeg -y -i ${folder}/train_00.mp4 -i ${folder}/train_01.mp4 -i ${folder}/train_02.mp4 -i ${folder}/train_03.mp4 -i ${folder}/train_04.mp4 -i ${folder}/train_04.mp4 -filter_complex "[0][1][2]hstack=inputs=3[v1];[3][4][5]hstack=inputs=3[v2];[v1][v2]vstack=inputs=2[vout]" -map "[vout]" -crf 10 ${folder}/train_comp.mp4

# render_test_020000_r_00716_fg.mp4
# render_test_020000_r_00716_fg_depth.mp4
# render_test_020000_r_00716_.mp4
# render_test_020000_r_00383_fg.mp4
# render_test_020000_r_00383_.mp4
# render_test_020000_r_00383_fg_depth.mp4
# render_test_020000_r_00050_.mp4
# render_test_020000_r_00050_fg_depth.mp4
# render_test_020000_r_00050_fg.mp4

