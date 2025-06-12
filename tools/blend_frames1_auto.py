#!/usr/bin/env python3
import os
from os import path
import sys
import subprocess
from glob import glob


def lcs(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        for j in range(len2):
            lcs_temp=0
            match=''
            while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and   string1[i+lcs_temp] == string2[j+lcs_temp]):
                match += string2[j+lcs_temp]
                lcs_temp+=1
            if (len(match) > len(answer)):
                answer = match

    return answer

dry = False
# dry = True
# inpstr = '../logs_auto/mlp_24-04-30_200_205_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t*_teps3e-2'
# inpstr = '../logs_auto/mlp_24-04-29_100_105_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t*_teps3e-2'
# inpstr = '../logs_auto/mlp_24-04-30_858_868_5fps_ls0.5_3e-2_newcode_lambdaanneal30k_1e-2_nolrsched_t*_teps3e-2'
# inpstr = '../logs_auto/cvprmultiseg_24-04-30a_207_168_5fps_ls0.5_3e-2_*'
inpstr = '../logs_auto/cvprmultiseg_24-04-30_200_205_5fps_ls0.5_3e-2_*'
# for split in ['', 'fg', 'fg_depth']:
# for split in ['', 'fg']:
# for split in ['fg_depth']:
for split in ['']:
    # split = ''
    # split = 'fg'
    # split = 'fg_depth'
    seqs = glob(inpstr)
    # seqs.sort(key=lambda x:int(x.split('d_t')[1].split('-')[0])) # todo: this is a hack
    seqs.sort(key=lambda x:int(x.split('3e-2_')[1].split('_')[0])) # todo: this is a hack
    # print(*seqs, sep='\n')
    # 1/0
    files = []

    answer = path.basename(seqs[0])
    for idx, seq in enumerate(seqs):
        # renders = [x for x in os.listdir(seq) if 'render' in x and 'bt' in x and path.isdir(path.join(seq, x))]
        renders = [x for x in os.listdir(seq) if 'render' in x and '5view' in x and path.isdir(path.join(seq, x))]
        # if idx == 0:
        #     renders = [x for x in renders if 'start' in x]
        # elif idx == len(seq)-1:
        #     renders = [x for x in renders if 'end' in x]
        # print(seq, renders)
        renders.sort()
        if renders:
            # files.append(path.join(seq, renders[-1], f'r_?????_{split}_????.png'))
            files.append(path.join(seq, renders[-1], f'????????????????????????_{split}_????.png'))

        answer = lcs(answer, path.basename(seq))
        pass
    print(seqs)
    # print(answer)
    print(files)
    if split:
        split = '_'+split
    out = answer+f'{split}.mp4'
    # print(' '.join(['python', 'blend_frames1.py']+sum([['-i', f] for f in files],[])+['-o', out]))
    if dry:
        pass
    else:
        subprocess.check_call(['python', 'blend_frames1.py', '-i'] + files +['-o', out])
