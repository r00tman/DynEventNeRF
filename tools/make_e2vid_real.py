#!/usr/bin/env python3
import os
from os import path
import sys
import subprocess

SCRIPT_DIR = path.dirname(sys.argv[0])

CONVERT_NPZ_BIN = path.join(SCRIPT_DIR, 'e2vid', 'convert_npz.py')
E2VID_TO_SERIAL_BIN = path.join(SCRIPT_DIR, 'e2vid', 'e2vid_to_serial.py')

E2VID_DIR = '/CT/EventNeRF/work/dynamic/rpg_e2vid'
E2VID_BIN = path.join(E2VID_DIR, 'run_reconstruction.py')


inpfn = sys.argv[1]

print('converting npz to zip...')
ts_per_sec = sys.argv[2]
t_min = sys.argv[3] if len(sys.argv) > 3 else '-1'
t_max = sys.argv[4] if len(sys.argv) > 4 else '-1'
subprocess.check_call(['python', CONVERT_NPZ_BIN, inpfn, ts_per_sec, t_min, t_max])

basename = path.splitext(path.basename(inpfn))[0]
zipfn = basename+'.zip'

print('running e2vid...')
# rm -r output/nextgen_r output/nextgen_r_serial
subprocess.check_call(['python', E2VID_BIN,
  '-c', path.join(E2VID_DIR, 'pretrained', 'E2VID_lightweight.pth.tar'),
  '-i', zipfn,
  '--color',
  '--display',
  '--show_events',
  '--fixed_duration',
  # '-T', '9.99',
  # '-T', '30.00',
  # '-T', '50.00',
  '-T', '49.00',
  '--Imin', '0',
  '--output_folder', 'output',
  '--dataset_name', basename])

print('converting results to serial frames...')
subprocess.check_call(['python', E2VID_TO_SERIAL_BIN, path.join('output', basename)])
# cd output
# ./convert.py nextgen_r

