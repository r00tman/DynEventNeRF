#!/usr/bin/env bash

# cameras=(0 1 2 3 4 5)
cameras=(0)
# export file="../rec/23-12-21/captury-calibra-2023_12_21_11_13_33.aedat4"
# export file="../rec/23-12-21/rec-2023_12_21_11_26_13.aedat4"
# export file="../rec/24-01-18/dvSaveExt-calib-2024_01_18_18_12_42.aedat4"
export file="../rec/24-01-18/dvSaveExt-rec-2024_01_18_18_41_00.aedat4"

detect_t0s() {
for cam in "${cameras[@]}"; do
    ./edi3 --detect_t0 -c ${cam} -i $file | grep t0 | cut -f2 -d' ' | cut -c4-
done
}

set -x
# export commont0=$(detect_t0s | sort | tail -n1)
# export commont0=1703154373201968
# export commont0=$(echo 46 1000000 * 1703154419201968 + p | dc)
export commont0=1705599695553854
echo commont0=$commont0

# export startT=$(echo 17055996955538543 $commont0 - 1000000 / p | dc)
# echo startT=$startT
export startT=158
# export duration=$(./edi3 --t0=${commont0} -c 0 --dry_run -i $file | grep "last event" | cut -f3 -d' ')
# echo duration=$duration
export endT=168
export fps=25
# exit


max_processes=2

# 1
# printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.7 -n -0.7 -a ${startT} -b ${endT} --t0=${commont0} -r ${fps} -e 12e-3 -s srgb -c {} -o rec-2024_01_18_18_41_00_{}_vng.mp4 -i ${file} --debayering vng"
# 2
# printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.7 -n -0.7 -a ${startT} -b ${endT} --t0=${commont0} -r ${fps} -e 12e-3 -s linear -c {} -o rec-2024_01_18_18_41_00_{}_vng.mp4 -i ${file} --debayering vng"
# 3
# printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.3 -n -0.3 -a ${startT} -b ${endT} --t0=${commont0} -r ${fps} -e 0e-3 -s linear -c {} -o rec-2024_01_18_18_41_00_{}_vng.mp4 -i ${file} --debayering vng"
# 4
# printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.2 -n -0.2 -a ${startT} -b ${endT} --t0=${commont0} -r ${fps} -e 0e-3 -s linear -c {} -o rec-2024_01_18_18_41_00_{}_vng.mp4 -i ${file} --debayering vng"
# 5
# printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.3 -n -0.4 -a ${startT} -b ${endT} --t0=${commont0} -r ${fps} -e 0e-3 -s linear -c {} -o rec-2024_01_18_18_41_00_{}_vng.mp4 -i ${file} --debayering vng"
# 6
# printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.3 -n -0.5 -a ${startT} -b ${endT} --t0=${commont0} -r ${fps} -e 0e-3 -s linear -c {} -o rec-2024_01_18_18_41_00_{}_vng.mp4 -i ${file} --debayering vng"


# lowest good offset for c0 = 10
cameras=(0 1 2 3 4 5)
# cameras=(1)
for off in $(seq 0 5 40); do
    # off=34000
    echo $off
    newct=$(echo $off 1000 '*' $commont0 + p | dc)
    echo $commont0
    echo $newct
    echo
    mkdir -p offsets/$off
    printf "%s\n" "${cameras[@]}" | xargs -n 1 -P $max_processes -I {} bash -c "./edi3 -p 0.3 -n -0.5 -a ${startT} -b ${endT} --t0=${newct} -r ${fps} -e 0e-3 -s linear -c {} -o offsets/$off/rec-2024_01_18_18_41_00_{}_vng.mp4 -i ${file} --debayering vng"
    ./compose.sh offsets/$off/rec-2024_01_18_18_41_00 _vng
done
