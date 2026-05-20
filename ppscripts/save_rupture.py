#!/usr/bin/env python

from ppscripts.save_ruptures import *
import ppscripts.postprocess as pp

sname = sys.argv[1]#'stick_break_{}'.format(sys.argv[1])

input_data = pp.get_input_data(sname)  

d=input_data['dc']/1000.0#d=input_data['nuc_xc0']/2.0/100.0
print('nuc_xc0',d)

save_ruptures([sname],d_slip=d)

dc=input_data['dc']
save_ruptures([sname],d_slip= dc, rpt_fname='rupture_min_stress.txt')
