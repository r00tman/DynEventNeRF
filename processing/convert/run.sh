#!/usr/bin/env bash
set -x
# ./run.py -o output/seg_59_64 -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 59 -e 64 -b 109
# ./run.py -o output/seg_146_156 -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 146 -e 156 -b 109
# ./run.py -o output/seg_292_302 -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109

# ./run.py -o output/oldroom_46_55 -c ../realdatapush/3_convertcalib/cameras.calib -a ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4 --t0 1703154373201968 -s 46 -e 55 -b 10 -r 1

# newgen 02-04-2024
# ./run.py -o output/oldroom_46_55_2 -c ../realdatapush/3_convertcalib/cameras.calib -a ../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4 --t0 1703154373201968 -s 46 -e 55 -b 10 -r 1

# ./run.py -o output/seg_59_64_1 -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 59 -e 64 -b 109
# ./run.py -o output/seg_146_156_1 -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 146 -e 156 -b 109
# ./run.py -o output/seg_292_302_1 -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109

# ./run.py -o output/seg_292_302_25fps -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109 -r 25
# ./run.py -o output/seg_292_302_5fps -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109 -r 5
# ./run.py -o output/seg_292_302_10fps -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109 -r 10

# ./run.py -o output/seg_292_302_5fps_linear0.32/train -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109 -r 5
# ./run.py -o output/seg_292_302_5fps_linearshift0.32_3e-2/train -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109 -r 5
# ./run.py -o output/seg_292_302_5fps_linearshift0.5_3e-2/train -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 292 -e 302 -b 109 -r 5

# ./run.py -o output/seg_59_64_5fps_ls0.5_3e-2/train -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 59 -e 64 -b 109 -r 5
# ./run.py -o output/seg_146_156_5fps_ls0.5_3e-2/train -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 146 -e 156 -b 109 -r 5

# ./run.py -o output/seg_370_380_5fps_ls0.5_3e-2/train -c input/cameras_event.calib -a ../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_33_17.aedat4 --t0 1705599197839417 -s 370 -e 380 -b 109 -r 5
#
# ./edi3 --detect_t0 -i ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4
# exit
# ./run.py -o output/24-04-29_150_160_5fps_ls0.5_3e-2/train -c input/2024-04-29/cameras.calib -a ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4 --t0 1714406790288158 -s 150 -e 160 -b 110 -r 5
# ./2_extract_mvframe.py ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4 1714406790288158 90 .
#
# ./run.py -o output/24-04-29_100_105_5fps_ls0.5_3e-2/train -c input/2024-04-29/cameras.calib -a ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4 --t0 1714406790288158 -s 100 -e 105 -b 110 -r 5
# ./run.py -o output/24-04-30_200_205_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 5
# ./run.py -o output/24-04-30_858_868_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 858 -e 868 -b 10 -r 5



# all remaining sequences
# ./run.py -o output/24-04-29_59_69_5fps_ls0.5_3e-2/train -c input/2024-04-29/cameras.calib -a ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4 --t0 1714406790288158 -s 59 -e 69 -b 110 -r 5
# ./run.py -o output/24-04-29_114_124_5fps_ls0.5_3e-2/train -c input/2024-04-29/cameras.calib -a ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4 --t0 1714406790288158 -s 114 -e 124 -b 110 -r 5
# ./run.py -o output/24-04-29_165_170_5fps_ls0.5_3e-2/train -c input/2024-04-29/cameras.calib -a ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4 --t0 1714406790288158 -s 165 -e 170 -b 110 -r 5

# ./run.py -o output/24-04-30_80_90_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 80 -e 90 -b 10 -r 5
# ./run.py -o output/24-04-30_256_266_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 256 -e 266 -b 10 -r 5
# ./run.py -o output/24-04-30_314_319_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 314 -e 319 -b 10 -r 5
# ./run.py -o output/24-04-30_332_342_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 332 -e 342 -b 10 -r 5
# ./run.py -o output/24-04-30_438_448_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 438 -e 448 -b 10 -r 5

# ./run.py -o output/24-04-30_682_687_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 682 -e 687 -b 10 -r 5
# ./run.py -o output/24-04-30_567_577_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 682 -e 687 -b 10 -r 5

# ./run.py -o output/24-04-30a_117_127_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_17_02_49.aedat4 --t0 1714489369704426 -s 117 -e 127 -b 10 -r 5
# ./run.py -o output/24-04-30a_163_168_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_17_02_49.aedat4 --t0 1714489369704426 -s 163 -e 168 -b 10 -r 5
# ./run.py -o output/24-04-30a_207_168_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_17_02_49.aedat4 --t0 1714489369704426 -s 207 -e 217 -b 10 -r 5

# patchup
# ./run.py -o output/24-04-30_567_577_5fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 567 -e 577 -b 10 -r 5


# ./run_blurry.py -o output/24-04-30_200_205_5fps_ls0.5_3e-2/train_blurry -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 5
# ./run_blurry.py -o output/24-04-30_314_319_5fps_ls0.5_3e-2/train_blurry -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 314 -e 319 -b 10 -r 5
# ./run_blurry.py -o output/24-04-30a_207_168_5fps_ls0.5_3e-2/train_blurry -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_17_02_49.aedat4 --t0 1714489369704426 -s 207 -e 217 -b 10 -r 5

# ./run_blurry.py -o output/24-04-30_80_90_5fps_ls0.5_3e-2/train_blurry -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 80 -e 90 -b 10 -r 5



# ./run.py -o output/24-04-30_200_205_1fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 1 --no_calib
# ./run.py -o output/24-04-30_200_205_2fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 2 --no_calib
# ./run.py -o output/24-04-30_200_205_10fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 10 --no_calib
# ./run.py -o output/24-04-30_200_205_20fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 20 --no_calib
# ./run.py -o output/24-04-30_200_205_30fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 30 --no_calib
# ./run.py -o output/24-04-30_200_205_50fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 50 --no_calib

# ./run.py -o output/24-04-30_200_205_100fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 100 --no_calib





# ./run.py -o output/24-04-30_200_205_200fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 200 -e 205 -b 10 -r 200
# ./run.py -o output/24-04-30_314_319_200fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 314 -e 319 -b 10 -r 200
./run.py -o output/24-04-30_80_90_400fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_16_47_30.aedat4 --t0 1714488450074116 -s 80 -e 90 -b 10 -r 100
./run.py -o output/24-04-30a_207_168_400fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_17_02_49.aedat4 --t0 1714489369704426 -s 207 -e 217 -b 10 -r 100
