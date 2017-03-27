#-*-encoding=utf-8-*-
#Program drawing psnr for each step 
#PSNR is recorded in psnr.log

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


if len(sys.argv) <2:
    fn = 'psnr.log'
else:
    fn = sys.argv[1]

f = open(fn, 'r')

r_dat = f.readlines()

step = []
dat = []
#step_re = re.compile('step (\d+),')
psnr_re = re.compile('PSNR: (.?\d+\.?\d*)')
for i, ln in enumerate(r_dat):
    #step_ = step_re.findall(ln)
    psnr = psnr_re.findall(ln)
    #step.append(step_[0])
    if len(psnr) > 0: 
        step.append(i)
        dat.append(psnr[0])

ds = pd.Series(index=step, data=dat)

plt.ylim([10, int(float(ds.max()))+1])
#plt.xticks(np.linspace(0, len(step), len(step)))
plt.plot(ds, marker='*')
plt.show()

