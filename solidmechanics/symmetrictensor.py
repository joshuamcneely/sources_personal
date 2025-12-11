# symmetrictensor.py
#
# Symmetric Tensor
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

import warnings
import numpy as np
from math import cos, sin, radians, pi

class SymmetricTensor(np.ndarray):
    """ Second order tensor 

    SymmetricTensor inherits from ndarray

    Attributes:
       self (numpy.array)           : tensor
       dim (int)                    : dimension of the matrix

    """

    def __new__(cls,x=[[0,0],[0,0]]):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(x, dtype=float).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
    
    def __init__(self,*args):
        """  Creates SymmetricTensor object.

        Args:
           tensor (array or list) : as array or list 

        Raises:
           RuntimeError: If the arguments tensor is not squared
           RuntimeError: If the arguments tensor is not symmetric
 
        """
        
        if not self.is_square():
            print(self)
            raise RuntimeError('tensor is not squared!!')
        if not self.is_symmetric():
            print(self)
            raise RuntimeError('tensor is not symmetric!!')
        
        self.dim = self.shape[0]


    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, SymmetricTensor):
            return np.max(self - other) <= 1e-12*np.max(np.abs(self))
        return False
    
    def __ne__(self, other):
        """Overrides the default implementation"""
        x = self.__eq__(other)
        if x is NotImplemented:
            return NotImplemented
        return not x

    def is_square(self):
        """Verify if an array is square

        Return:
            bool: True if succesful
        """
        if self.shape[0] == self.shape[1]:
            return True 
        else:
            return False

    def is_symmetric(self):
        """Verify if an array is symmetric

        Return:
            bool: True if succesful
        """
        return self == self.T
        
    def make_symmetric(self):
        """Make an array symmetric
        """
        self[:,:] = (self + self.T)/2.0 
        
            
    def eigen_values(self):
        """compute eigenvalues

        Args:
            self

        Return: 
            SymmetricTensor object. The tensor is diagonal with the eigenvalues in the diagonal
        """
        return self.eigen_values_vectors()[0]
    
    def eigen_values_vectors(self):
        """
        Args:
            self

        Return: 
            w: SymmetricTensor object. The tensor is diagonal with the eigenvalues in the diagonal
            v: SymmetricTensor object. The tensor with eigenvectors stored in colums v[:i] is the igenvector corresponding to eigenvalues w[i,i]
        """
        
        eigenValues, eigenVectors = np.linalg.eig(self)

        # sort
        idx = eigenValues.argsort()[::-1]   
        w = SymmetricTensor(np.diag(eigenValues[idx]))
        v = eigenVectors[:,idx]
        return w,v

    def rotate(self,angle):
        """Rotate tensor

        Args:
           angle(float) : value of the rotation angle in radian
        
        Return:
           SymmetricTensor object with a rotated tensor

        Raises:
           RuntimeError: if the tensor is not of two dimensions
           
        """
        if not self.shape[0] == 2:
            print('Tensor is not of two dimensions!\nnot implemented in 3d')
            raise RuntimeError
        
        #angle = radians(angle)

        if angle > 2*pi:
            warnings.warn('angle does not look like being in radian!')

        c = np.matrix([[cos(angle), sin(angle)],
                       [-sin(angle), cos(angle)]])

        return SymmetricTensor(np.matmul(np.matmul(c,self),c.T))

    def get_invariants(self):
        i1 = np.trace(self)
        i2 = 0.5*(i1**2 - np.trace(np.dot(self,self)))
        i3 = np.linalg.det(self)
        return i1,i2,i3


def test_symmetrictensor():

    try:
        st =  SymmetricTensor([[1,2],[3]])
    except:
        pass
    else:
        raise RuntimeError('should fail because non-square')

    try:
        st  =  SymmetricTensor([[0,0],[1,0]])
    except:
        pass
    else:
        raise RuntimeError('should fail because non-symmetric')
    try:
        st  =  SymmetricTensor([[1,0],[1e-16,2]])
    except:
        raise RuntimeError('should not fail because symmetric (w/ num. error)')
    else:
        pass

    # don't know what this is for
    #d00=np.array([1,2,3,4])
    #d01=np.array([5,6,7,8])
    #d11=np.array([9,10,10,11])
    #tensor3d = np.array([[d00,d01],[d01,d11]])
    #print(tensor3d)
    #print(tensor3d.shape)#(2,2,4)
    #print(tensor3d[0,0])
    #print(tensor3d.T) 
    #print(tensor3d.T.shape) #( 4,2,2)
    #symtensors=[SymmetricTensor(i) for i in tensor3d.T]

    ### test eig
    tmp = np.random.random((3,3))
    A = (tmp + tmp.T)/2.0 # make symmtric

    eigenValues,eigenVectors = np.linalg.eig(A)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = np.diag(eigenValues[idx])
    eigenVectors = eigenVectors[:,idx]

    testA = SymmetricTensor(A)
    w0 = testA.eigen_values()
    w1,v1 = testA.eigen_values_vectors()

    for t1,t2 in zip([w0,w1,v1],[w1,eigenValues,eigenVectors]):
        if not np.array_equal(t1,t2):
            print(t1,'\n',t2)
            raise RuntimeError('eigen test failed')

    # test invariants
    testA = SymmetricTensor(A)
    i1,i2,i3 = testA.get_invariants()

    if abs(i1 - np.trace(A)) > 1e-12:
        print(i1)
        print(np.trace(A))
        raise RuntimeError('i1 wrong')

    i2_2=A[0,0]*A[1,1] + A[1,1]*A[2,2] + A[0,0]*A[2,2]-A[0,1]*A[0,1]-A[1,2]*A[2,1]-A[0,2]*A[2,0]
    if abs(i2 - i2_2)>1e-12:
        print(i2)
        print(i2_2)
        raise RuntimeError('i2 wrong')

    if abs(i3 - np.linalg.det(A)) > 1e-12:
        print(i3)
        print(np.linalg.det(A))
        raise RuntimeError('i3 wrong')

# ---------------------------------------------------------
if (__name__ == '__main__'):
    print('unit tests: symmetric tensor')
    test_symmetrictensor()
    print('success!')
