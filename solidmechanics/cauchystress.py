# cauchystress.py
#
# Cauchy Stress Tensor
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

import numpy as np

from .symmetrictensor import SymmetricTensor

class CauchyStress(SymmetricTensor):
    pass

    def principal_stresses(self):
        """ 
        Args:
            self

        Return: 
            CauchyStress object. The tensor is diagonal with the principal stresses in the diagonal
        """
        return CauchyStress(self.eigen_values())
    
    def principal_stresses_directions(self):
        """ 
        Args:
            self

        Return: 
            w: CauchyStress object. The tensor is diagonal with the principal stresses in the diagonal
            v: np.array object. The tensor with eigenvectors (directions) stored in colums v[:i] is the igenvector corresponding to eigenvalues w[i,i]

        """
        w,v=self.eigen_values_vectors()
        return CauchyStress(w),v
        
    def surface_traction(self,normal):
        """
        Args:
            self
            normal (np.array): 1d array
            
        Return:
            traction (np.array):
        """
        n = normal/np.linalg.norm(normal)
        return np.asarray(np.dot(self,n))
    
        
    def get_deviatoric_stress(self):
        Sij = CauchyStress()
        Sij = self - np.identity(self.shape[0])*np.trace(self)/3.0
        return CauchyStress(Sij)

# -------------------------------------------------------------------
def test_cauchy():
    cs1 = CauchyStress([[3,2],
                        [2,3]])

    cs1ps = CauchyStress([[5,0],[0,1]]) # solution
    cs1p = cs1.principal_stresses()
    if cs1p != cs1ps:
        print(cs1p)
        print(cs1ps)
        print(cs1p-sc1ps)
        raise RuntimeError('principle stress is wrong')

    st1 = cs1.surface_traction([1,1])
    st1s = np.array([5/np.sqrt(2),5/np.sqrt(2)]) # solution
    if False in st1 == st1s:
        print(st1)
        print(st1s)
        raise RuntimeError('surface traction is wrong')

    import math
    cs1r = cs1.rotate(math.radians(45))
    cs1rs = CauchyStress([[5,0],[0,1]]) # solution
    if cs1r != cs1rs:
        print(cs1r)
        print(cs1rs)
        print(cs1r-sc1rs)
        raise RuntimeError('stress rotation is wrong')


# ----------------------------------------------
if (__name__ == '__main__'):
    print('unit tests: cauchy stress tensor')
    test_cauchy()
    print('success!')
