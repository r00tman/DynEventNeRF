#!/usr/bin/env bash

cameras=(0 1 2 3 4 5)
# export file="../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4"
export file="../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4"

detect_t0s() {
for cam in "${cameras[@]}"; do
    ./edi3 --detect_t0 -c ${cam} -i $file | grep t0 | cut -f2 -d' ' | cut -c4-
done
}

export commont0=$(detect_t0s | sort | tail -n1)

echo t0=$commont0
export duration=$(./edi3 --t0=${commont0} -c 0 --dry_run -i $file | grep "last event" | cut -f3 -d' ')
echo duration=$duration

max_processes=2

printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.7 -n -0.7 -a 0 -b ${duration} --t0=${commont0} -r 25 -e 12e-3 -s srgb -c {} -o calib_0707_{}_vng.mp4 -i ${file} --debayering vng"
