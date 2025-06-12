Multi-view event + RGB processing software
---

Overview:
- `edi` - fast deblurring, interpolation, debayering, tonemapping (fast deblurring is based on Lin et al., ICRA 2023 paper),
- `dvstat_cpp` - getting camera names from aedat4 (somehow not a feature of `dv` or `dv-processing` python packages),
- `convert` - converting multi-view aedat4 into a full dyneventnerf training dataset folder, needs both other packages to work.

Installation
---
`dvstat_cpp` needs `dv-processing` C++ library installed as dependency and available through `pkg-config`.

```bash
cd dvstat_cpp
python setup.py develop
```

`edi` needs `dv`, `dv-processing` and `libav` C++ libraries as dependencies available through `pkg-config` to compile.

```bash
cd edi
make
```

How to turn multi-view event+blurry RGB aedat4 into DynEventNeRF training data
---

Please use `convert/run.py`. We provide many examples in `convert/run.sh`. For example,
```bash
./run.py -o output/24-04-30a_207_168_400fps_ls0.5_3e-2/train -c input/2024-04-30/cameras.calib -a ../rec/24-04-30/dvSaveExt-rec-2024_04_30_17_02_49.aedat4 --t0 1714489369704426 -s 207 -e 217 -b 10 -r 100
```

This will save training data to ` output/24-04-30a_207_168_400fps_ls0.5_3e-2/train` using
- Captury camera calibration `input/2024-04-30/cameras.calib`,
- Input multi-view event+RGB file `../rec/24-04-30/dvSaveExt-rec-2024_04_30_17_02_49.aedat4`,
- Common t=0 for all views at `1714489369704426` (to synchronize the views),
- Start at 207 seconds and end at 217 seconds since `t0=1714489369704426`,
- Background RGBs captured at `t=10s`,
- Generated frame rate of 100 reference RGB's per second (using EDI interpolation).

To extract `t0` values from multi-view aedat recording, you could use `edi3`:
```bash
./edi3 --detect_t0 -i ../rec/24-04-29/dvSaveExt-rec-2024_04_29_18_06_30.aedat4
```
This will print t0's for all views. From there you can choose which one would be the main view and use its t0. If possible, pick the one that minimizes the amount of accumulation needed for all views. This can improve the quality of deblurring and interpolation.

Converting multi-view event+blurry RGB aedat4 into synchronized sharp calibration videos for use Captury/etc
---
This is how to turn multi-view aedat4 recordings into synced higher-res calibration videos 

For EDI, we used these camera parameters: `-p 0.5 -n -0.5 -e 3e-2 -s linearshift --debayering vng`.

`edi/process5.sh` is an example of how to do `aedat4->synced multi-view mp4's` and probably is a good starting point.

Note, that Captury and many other software don't work with `yuv444` videos, i.e. when each pixel has its own R, G, and B values. They instead prefer `yuv420`, where color information is downsampled. Since our resolution is small (346x260), directly converting to `yuv420` would make results blurry and the calibration would likely fail. To fix this, we used ffmpeg to upscale `yuv444` 346x260 mp4's into `yuv420` 692x520 avi's, so no information is lost:
```bash
ffmpeg -i input.mp4 -vf "scale=692:520" -pix_fmt yuv420p -crf 10 output.avi
```

