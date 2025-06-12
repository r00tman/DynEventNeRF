#!/usr/bin/env bash
A=()
for i in $(seq $2); do
    A+=(-i $1)
done

echo running: ffmpeg "${A[@]}" -filter_complex "concat=n=$2" -crf 10 ${1%%.mp4}_loop${2}x.mp4
ffmpeg "${A[@]}" -filter_complex "concat=n=$2" -crf 10 ${1%%.mp4}_loop${2}x.mp4
