# infinitesimalstrain.py
#
# Infinitesimal Strain Tensor
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @author Chun-Yu Ke <ck659@cornell.edu>
# @date     2015/10/01
# @modified 2020/12/12

from __future__ import print_function, division, absolute_import

from .symmetrictensor import SymmetricTensor

class InfinitesimalStrain(SymmetricTensor):
    pass

def test_infinitesimalstrain():
    pass

# ---------------------------------------------------------
if (__name__ == '__main__'):
    print('unit tests: infinitesimal strain')
    test_infinitesimalstrain()
    print('nothing done')
