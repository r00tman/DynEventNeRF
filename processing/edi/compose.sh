#!/usr/bin/env bash
prefix=$1
suffix=$2
ffmpeg -y -i ${prefix}_0${suffix}.mp4 -i ${prefix}_1${suffix}.mp4 -i ${prefix}_2${suffix}.mp4 -i ${prefix}_3${suffix}.mp4 -i ${prefix}_4${suffix}.mp4 -i ${prefix}_5${suffix}.mp4 -filter_complex "[0][1][2]hstack=inputs=3[v1];[3][4][5]hstack=inputs=3[v2];[v1][v2]vstack=inputs=2[vout]" -map "[vout]" -crf 10 ${prefix}_comp${suffix}.mp4
ffmpeg -y -i ${prefix}_0${suffix}_deblur.mp4 -i ${prefix}_1${suffix}_deblur.mp4 -i ${prefix}_2${suffix}_deblur.mp4 -i ${prefix}_3${suffix}_deblur.mp4 -i ${prefix}_4${suffix}_deblur.mp4 -i ${prefix}_5${suffix}_deblur.mp4 -filter_complex "[0][1][2]hstack=inputs=3[v1];[3][4][5]hstack=inputs=3[v2];[v1][v2]vstack=inputs=2[vout]" -map "[vout]" -crf 10 ${prefix}_comp${suffix}_deblur.mp4
ffmpeg -y -i ${prefix}_0${suffix}_blur.mp4 -i ${prefix}_1${suffix}_blur.mp4 -i ${prefix}_2${suffix}_blur.mp4 -i ${prefix}_3${suffix}_blur.mp4 -i ${prefix}_4${suffix}_blur.mp4 -i ${prefix}_5${suffix}_blur.mp4 -filter_complex "[0][1][2]hstack=inputs=3[v1];[3][4][5]hstack=inputs=3[v2];[v1][v2]vstack=inputs=2[vout]" -map "[vout]" -crf 10 ${prefix}_comp${suffix}_blur.mp4
