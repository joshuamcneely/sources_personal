#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np
from collections import Iterable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d

import ifasha.datamanager as idm

wdir = './data'
bname = 'dynamic_3d_taui4.60e+06_nucC0.0HS5.0e-02WZ5.00e-03V1400_sigi5.00e+06_E5.65e+09nu0.33rho1180_lswnhs0.94k0.74d2.24e-06_L0.12H0.07W0.005_elem960'
bname = 'dyn_3d_TE5.65e+09nu0.33rho1180_lswnhs0.94k0.74d2.24e-06_taui4.60e+06sigi5.00e+06_nucC0.0HS5.0e-02WZ5.00e-03V1400_L0.2Ht0.1Wt0.0025zsymmsh1600'
bname = 'dyn_3d_TE5.65e+09nu0.33rho1180_lswnhs0.94k0.74d2.24e-06_taui4.30e+06sigi5.00e+06_nucC0.0HS5.0e-02WZ1.30e-02V1400_L0.2Ht0.1Wt0.0025zsymmsh1600'
base = "3d_taup1_lowerprestress"

#end_time = 1.3e-4

# ---------------------------------------------------------------------
# full
flds = [[idm.FieldId('strain',0), "eps11"],
        [idm.FieldId('strain',1), "eps22"],
        [idm.FieldId('strain',2), "eps12"]]
group = 'full'

# at interface
flds = [
    [[idm.FieldId('friction_traction',0), idm.FieldId('friction_traction',1)], 'ft'], # inclined interface
#    [[idm.FieldId('contact_pressure',0), idm.FieldId('contact_pressure',1)], 'cp'],
#    [idm.FieldId('slip',0), 'slip'],
    [idm.FieldId('slip_velocity',0), 'slip_velocity']
]
#flds = [[idm.FieldId('friction_traction',0), 'ft']] # horizontal interface
group = 'interface'

# off fault
flds = [[idm.FieldId('strain',0), "eps11"],
        [idm.FieldId('strain',1), "eps22"],
        [idm.FieldId('strain',2), "eps12"]]
group = 'at-distance'
#group = 'at-dist-7.5mm'
#group = 'at-dist-8.0mm'

# ---------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)

# ---------------------------------------------------------------------
# load the simulation data
dma = idm.DataManagerAnalysis(bname,wdir)
data = dma(group)

# find points on a z coordinate
z_coords = sorted(set(data.get_field_at_t_index(idm.FieldId('position',2),0)[0]))
#z_select = z_coords[len(z_coords)/2]
z_select = z_coords[-1]
#z_select = z_coords[0]
idcs = data.get_indices_of_nodes_on_line(idm.FieldId('position',2),z_select)


simid = data.sim_info.split('=')[1]
name = base+"_"+simid+"_"+'z{:1.2e}'.format(z_select)
print(name)

positions_x = []
positions_y = []
positions_z = []
fields_data = []
for fld in flds:
    fields_data.append([])

times = []
#lasttidx = data.get_index_of_closest_time(idm.FieldId('time'),end_time)
lasttidx = data.get_t_index('last')
print('last t index =',lasttidx)
for tidx in range(lasttidx):

    times.append(data.get_field_at_t_index(idm.FieldId('time'),tidx)[0])

    position_x = data.get_field_at_t_index(idm.FieldId('position',0),tidx)
    position_y = data.get_field_at_t_index(idm.FieldId('position',1),tidx)
    position_z = data.get_field_at_t_index(idm.FieldId('position',2),tidx)

    position_x = position_x[0,idcs]
    position_y = position_y[0,idcs]
    position_z = position_z[0,idcs]


    fltr       = position_x.argsort()
    position_x = position_x[fltr]
    positions_x.append(position_x[:])

    position_y = position_y[fltr]
    positions_y.append(position_y[:])

    position_z = position_z[fltr]
    positions_z.append(position_z[:])
    
    for f in range(len(flds)):
        fld = flds[f]
        print(tidx,fld[0])

        if not isinstance(fld[0], Iterable): # only one field
            field = data.get_field_at_t_index(fld[0], tidx)[0,idcs]
            fields_data[f].append(field[fltr])
         
        else:
            field_to_add = 0
            for ff in fld[0]:
                field = data.get_field_at_t_index(ff, tidx)[0,idcs]
                field_to_add += field * field
            field_to_add = np.sqrt(field_to_add)
            fields_data[f].append(field_to_add[fltr])
            
    
times       = np.array(times)    
positions_x = np.transpose(np.array(positions_x))
positions_y = np.transpose(np.array(positions_y))
positions_z = np.transpose(np.array(positions_z))
for fd in fields_data:
    fd = np.transpose(np.array(fd))

np.savetxt(name+"_"+group+"_time"+".csv", times, delimiter=",")
np.savetxt(name+"_"+group+"_x"+".csv", positions_x[:,0], delimiter=",")
np.savetxt(name+"_"+group+"_y"+".csv", positions_y[:,0], delimiter=",")
np.savetxt(name+"_"+group+"_z"+".csv", positions_z[:,0], delimiter=",")
for f in range(len(fields_data)):
    np.savetxt(name+"_"+group+"_"+flds[f][1]+".csv", fields_data[f], delimiter=",")
