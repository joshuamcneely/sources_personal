#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import os.path
import shutil as shtl
import sys

from DataManager import FieldId
from DataManager import Field

flds = ['position_0 3201 10 float64 N verified_position_0.mmp.original',
        'friction_traction_0 3201 10 float64 N verified_friction_traction_0.mmp.original']

nbts = 10

for fld in flds:
    
    src_fld = Field()
    dst_fld = Field()
    
    src_fld.setPath('.')
    dst_fld.setPath('.')

    src_fld.loadString(fld)

    fid = FieldId()
    fid.loadString('verified_'+fld.split()[0])

    dst_fld.loadString(fld)
    dst_fld.identity = fid
    dst_fld.createFileName()

    src_mmap = src_fld.getMemMap()
    dst_mmap = dst_fld.getMemMap('w+')

    for (i,j), value in np.ndenumerate(src_mmap):
        dst_mmap[i,j] = value

