#!/usr/bin/env python3

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt

import solidmechanics as sm
import solidmechanics.lefm as lefm

mat = sm.LinearElasticMaterial({
    sm.smd.E : 5.65e9,
    sm.smd.nu : 0.33,
    sm.smd.rho: 1180,
    sm.smd.pstress: True
})

iface = {sm.smd.Gamma : 1}

v0 = 0.7 * mat[sm.smd.cR]

# area and discretization of interest
D = 0.005
N = 101
x,y = np.meshgrid(np.linspace(-D,D,N),
                  np.linspace(-D,D,N))

# LEFM solution for singular crack in area of interest
Sxx,Syy,Sxy = lefm.mode_2_subRayleigh_singular_stress_field(v0, mat, iface,
                                                            x=x, y=y,
                                                            rformat='element-wise')
stresses=[Sxy,Sxx,Syy]
lbls=['Sxy','Sxx','Syy']

fig = plt.figure()
for S,i,lbl in zip(stresses,range(len(stresses)),lbls):
    ax = fig.add_subplot(3,1,i+1)

    Smax = np.nanmax(S[y != 0])
    Smin = np.nanmin(S[y != 0])
    R=0.7
    fg = ax.pcolormesh(x,y,S,
                       vmax=R*Smax,
                       vmin=R*Smin)
    cbar = fig.colorbar(fg,label=lbl)

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

