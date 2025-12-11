#!/usr/bin/env python3

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import numpy as np

import solidmechanics as sm
import solidmechanics.lefm as lefm

mat = sm.LinearElasticMaterial({
    sm.smd.E : 5.65e9,
    sm.smd.nu : 0.35,
    sm.smd.rho : 1170,
    sm.smd.pstrain : True
})

# coordinate
dx=0.0001
X=0.8
x = np.arange(0,X,dx)

# peak strength
tp = np.ones_like(x)*1e6
# fracture energy
Gamma = np.ones_like(x)

l0 = Gamma[0] * mat[sm.smd.mu] / tp[0]**2
print('l0',l0)

# pre-stress uniform (to compare with uniform solution)
t0 = np.ones_like(x) * 0.3e6

interface = {
    sm.smd.cohesive_zone_s : 'linear',
    sm.smd.Gamma : Gamma[0],
    sm.smd.tauc : tp[0],
    sm.smd.taur : 0,
}

fig = plt.figure()
ax = fig.add_subplot(111)

# all solutions should superpose on figure!!

# compute solution for uniform problem (as reference)
eom = lefm.supershear_equation_of_motion(1e5,
                                         mat,
                                         uni_Gamma=Gamma[0],
                                         uni_tau0=t0[0],
                                         uni_taup=tp[0],
                                         crack_type='bilateral',
                                         cohesive_zone_type='linear')
ax.plot(eom[:,0]/l0,
        eom[:,1]/mat[sm.smd.cp])

# compute uniform solution but only up to to maximal crack length provided (here 0.1254)
eom = lefm.supershear_equation_of_motion(np.array([0.0002496, 0.03, 0.1254]),
                                         mat,
                                         uni_Gamma=Gamma[0],
                                         uni_tau0=t0[0],
                                         uni_taup=tp[0],
                                         crack_type='bilateral',
                                         cohesive_zone_type='linear',
                                         nb_root_sec=20)
ax.plot(eom[:,0]/l0,
        eom[:,1]/mat[sm.smd.cp],'s')

# compute non-uniform solution for 4 points:
# - first too short (does not show up)
# - second where there is only one solution
# - third where there are three solutions
# - forth too far: beyond the max(x)
eom = lefm.supershear_equation_of_motion(np.array([0.0002496, 0.03, 0.1254,1e5]),
                                         mat,
                                         x=x,
                                         Gamma=Gamma,
                                         tau0=t0,
                                         taup=tp,
                                         crack_type='bilateral',
                                         cohesive_zone_type='linear',
                                         nb_root_sec=20)
ax.plot(eom[:,0]/l0,
        eom[:,1]/mat[sm.smd.cp],'o')

# check if it works if a single value is provided for crack length instead of list
eom = lefm.supershear_equation_of_motion(0.08,
                                         mat,
                                         x=x,
                                         Gamma=Gamma,
                                         tau0=t0,
                                         taup=tp,
                                         crack_type='bilateral',
                                         cohesive_zone_type='linear',
                                         nb_root_sec=20)

ax.plot(eom[:,0]/l0,
        eom[:,1]/mat[sm.smd.cp],'o')
    
ax.set_xlim([0,300])
ax.set_ylim([0.4,1])
ax.set_xlabel('x/l0')
ax.set_ylabel('v/Cp')

plt.show()
