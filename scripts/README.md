SLURM quick launcher
---
No more writing SLURM batch scripts or fiddling with commands. Submit exactly the same command you ran locally by just prepending `./launcher.sh`. It also allows easy per-run code archival and auto-restarts.

Installation
---
Change the log path and conda environment in `launcher_job.sh` from `<absolute-path-to-code>` and `<path-to-conda-env>` to your paths.

Usage
---
Launch 1hr job:
```
./launcher.sh <command> <arguments>
```

Launch 1hr job restarting itself 10 times:
```
./launcher_multi.sh <command> <arguments>
```

Launch 1hr job restarting itself 10 times AND archive all the code (reruns happen from the archived code, so you can modify original code without affecting the reruns):
```
./launcher_multi_archive.sh <command> <arguments>
```

Example
---
This will archive and launch training in 1hr jobs for 10 restarts.
```
./launcher_multi_archive.sh python ddp_train_nerf.py --config configs/deff/event10.txt
```


Running time (1hr) can be changed in `launcher_job.sh` and the number of restarts can be changed in `launcher_multi.sh`/`launcher_multi_archive.sh`
