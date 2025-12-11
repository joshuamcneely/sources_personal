#!/usr/bin/env python3

from __future__ import print_function, division, absolute_import

import numpy as np

from ifasha.rupturehunter.rupture_hunter import SlipPeriod, SlipPeriodsAtX
from ifasha.rupturehunter.rupture_hunter import SPAXGrid, SPAXCollection
from ifasha.rupturehunter.rupture_hunter import Rupture, RuptureHunter


# ---------------------------------     SlipPeriod

sp1 = SlipPeriod(2.3,5.6,77)
str1 =sp1.write_string() 

sp2 = SlipPeriod(None,None)
sp2.read_string(str1)

if not sp1 == sp2:
    print('SlipPeriods should be the same')
    print(sp1)
    print(sp2)
    raise RuntimeError

# ---------------------------------     

pos = np.array([[float(i) for i in range(5,0,-1)]])
print(pos)
tim = np.array([float(i) for i in range(20,25)])
iss = np.array([[0,1,1,1,1],
                [0,0,1,1,1],
                [1,0,0,1,1],
                [0,0,0,0,0],
                [1,1,1,1,1]])
print(pos.shape,tim.shape,iss.shape)

# rupture hunter to get rupture
hunter1 = RuptureHunter()
hunter1.load(pos,tim,iss)
hunter1.hunt()
rpt1 = hunter1.get_rupture(0)

# ---------------------------------     SPAXGrid 
# get string from grid (unsorted grid)
grid1 = rpt1.grid
str1 = grid1.write_string()

# have a second grid and read string
grid2 = SPAXGrid()
grid2.read_string(str1)

if not grid1 == grid2:
    print('The two grids should be the same!')
    print(grid1.write_string())
    print(grid2.write_string())
    raise RuntimeError

# sort the second grid
grid2.sort_grid()

if grid1 == grid2:
    print('The two grids should be different')
    print(grid1.write_string())
    print(grid2.write_string())
    raise RuntimeError

# first grid reads the new string (sorted grid)
str1 = grid2.write_string()
grid3 = SPAXGrid()
grid3.read_string(str1)

if not grid3 == grid2:
    print('The two grids should be the same!')
    print(grid2.write_string())
    print(grid3.write_string())
    raise RuntimeError


# ---------------------------------     SPAX
spax1 = rpt1.getSPAXAtX((5,0))
str1=spax1.write_string()

spax2 = SlipPeriodsAtX(None)
spax2.read_string(str1)

if not spax1.__repr__() == spax2.__repr__():
    print('SPAX should be the same')
    print(spax1)
    print(spax2)
    raise RuntimeError

spax3 = spax2.first()

if not len(spax3) == 1 and spax3[0] == spax2[0]:
    print('SPAX should have only first slip period')
    print(spax2)
    print(spax3)
    raise RuntimeError

# ---------------------------------     SPAXCollection (Rupture)

spaxcl = list()
spaxcl = rpt1.write_strings()

# make sure that these are all strings (as it would be coming from a file)
spaxcl = [str(i) for i in spaxcl]

#spaxc2 = SPAXCollection()
#spaxc2.read_strings(spaxcl)

rpt2 = Rupture()
rpt2.read_strings(spaxcl)

if not rpt2.__repr__() == rpt1.__repr__():
    print('Both ruptures should be the same')
    print(rpt1)
    print(rpt2)
    raise RuntimeError

rpt3 = rpt2.first()

for gp,s in rpt3.items():
    if len(s) != 1:
        print('all spaxs should have only one slip period')
        print(rpt2)
        print(rpt3)
        raise RuntimeError
