#!/usr/bin/env bash
folder=$1
ffmpeg -y -i ${folder}/test00050.mp4 -i ${folder}/test00383.mp4 -i ${folder}/test00716.mp4 -filter_complex "[0][1][2]hstack=inputs=3[vout]" -map "[vout]" -crf 10 ${folder}/test_comp.mp4
ffmpeg -y -i ${folder}/train_00.mp4 -i ${folder}/train_01.mp4 -i ${folder}/train_02.mp4 -i ${folder}/train_03.mp4 -i ${folder}/train_04.mp4 -i ${folder}/train_04.mp4 -filter_complex "[0][1][2]hstack=inputs=3[v1];[3][4][5]hstack=inputs=3[v2];[v1][v2]vstack=inputs=2[vout]" -map "[vout]" -crf 10 ${folder}/train_comp.mp4
