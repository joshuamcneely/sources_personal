# utilities.py
#
# useful functions
#
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2015/10/01
# @modified 2020/12/12

from __future__ import print_function, division, absolute_import

import numpy as np

def cartesian_to_polar(x,y):
    """Convert cartesian coordinates to polar

    Args:
       x(array): coordinate along the x axis
       y(array): coordinate along the y axis

    Return:
       r(array): distance from the origin
       theta(array): angle from a reference axis

    """

    x = np.array(x)
    y = np.array(y)

    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)

    return r, theta

def polar_to_cartesian(r,theta):
    """Convert polar coordinates to cartesian

    Args:
       r(array): distance from the origin
       theta(array): angle from a reference axis

    Return:
       x(array): coordinate along the x axis
       y(array): coordinate along the y axis

    """

    
    r = np.array(r)
    theta = np.array(theta)

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return x,y

def test_utilities():

    x = np.array([[1,-1],[-1,1]])
    y = np.array([[1,1],[-1,-1]])
    r, theta = cartesian_to_polar(x,y)
    xc, yc = polar_to_cartesian(r,theta)
    if False in x==xc or False in y==yc:
        import math
        print(r)
        print(theta/2/math.pi*360)
        print(xc)
        print(yc)
        raise RuntimeError('coordinate transformation is wrong')

# ---------------------------------------------------------
if (__name__ == '__main__'):
    print('unit tests: utilities')
    test_utilities()
    print('success!')
