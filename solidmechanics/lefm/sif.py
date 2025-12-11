# sif.py
#
# stress intensity factor
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2015/10/01
# @modified 2020/12/11

from __future__ import print_function, division, absolute_import

import numpy as np

from .functions import epsilon_bimat_const

def K1_static(a,config,**kwargs):
    """
    Compute static mode I stress intensity factor for various configurations

    Args:
      a (list) : crack half lengths
      config (str): description of geometric configuration

    Kwargs:
      matprop1 (dict): material property of top material
      matprop2 (dict): material property of bottom material

    Returns:
      K1 (list): mode I stress intensity factor
    """

    if config == "bilat_bimat_inf_domain":
        # this is only part of the complex SIF 
        # checkout Hutchinson, Mear & Rice (1987)
        # K = K1 + i*K2
        # here we do only K1
        tauinf = kwargs.get('tauinf',0.)
        siginf = kwargs.get('siginf',0.)
        eps = epsilon_bimat_const(kwargs.get('matprop1'),
                                  kwargs.get('matprop2'))
        # Rice & Sih (1965) eq. (31) 
        # w/o /cosh(pi*eps) see Hutchinson, Mear & Rice (1987) after eq. (2.2)
        k1  = siginf * (np.cos(eps*np.log(2*a)) + 2*eps*np.sin(eps*np.log(2*a)))
        k1 += tauinf * (np.sin(eps*np.log(2*a)) - 2*eps*np.cos(eps*np.log(2*a)))
        #k1 /= np.cosh(eps*np.pi)
        k1 *= np.sqrt(a)
        # Hutchinson, Mear & Rice (1987) after eq. (2.2)
        K1 = np.sqrt(np.pi) * k1
    else:
        print("Don't know configuration: {}".format(config))
        raise RuntimeError

    return K1


def K2_static(a,config,**kwargs):
    """
    Compute static mode II stress intensity factor for various configurations

    Args:
      a (list) : crack half lengths
      config (str): description of geometric configuration

    Kwargs:
      matprop1 (dict): material property of top material
      matprop2 (dict): material property of bottom material

    Returns:
      K2 (list): mode II stress intensity factor
    """

    if config == "bilat_inf_domain":
        tauinf = kwargs.get('tauinf')
        K2 = tauinf * np.sqrt(np.pi*a)
    elif config == "bilat_bimat_inf_domain":
        # this is only part of the complex SIF 
        # checkout Hutchinson, Mear & Rice (1987)
        # K = K1 + i*K2
        # here we do only K2
        tauinf = kwargs.get('tauinf',0.)
        siginf = kwargs.get('siginf',0.)
        eps = epsilon_bimat_const(kwargs.get('matprop1'),
                                  kwargs.get('matprop2'))
        # Rice & Sih (1965) eq. (31) 
        # w/o /cosh(pi*eps) see Hutchinson, Mear & Rice (1987) after eq. (2.2)
        k2  = tauinf * (np.cos(eps*np.log(2*a)) + 2*eps*np.sin(eps*np.log(2*a)))
        k2 -= siginf * (np.sin(eps*np.log(2*a)) - 2*eps*np.cos(eps*np.log(2*a)))
        #k2 /= np.cosh(eps*np.pi)
        k2 *= np.sqrt(a)
        # Hutchinson, Mear & Rice (1987) after eq. (2.2)
        K2 = np.sqrt(np.pi) * k2
    else:
        print("Don't know configuration: {}".format(config))
        raise RuntimeError

    return K2

# ----------------------------------------------
# test unimaterial set-up
# ----------------------------------------------
def test_sif_unimaterial():

    a0 = 1/(1e4*np.pi)
    tau8 = 1e6
    K2sol = 1e4
    
    K2st = K2_static(a0,'bilat_inf_domain',tauinf=tau8)

    if abs(K2st - K2sol) > 1e-6*K2sol:
        print(K2st)
        raise RuntimeError('Failed unimat K2')

# ----------------------------------------------
#test bimaterial set-up with twice same material
# ----------------------------------------------
def test_sif_bimaterial_1():

    from ..definitions import smd
    from ..linearelasticity import LinearElasticMaterial as lem
    mat1 = lem({smd.E : 2.00e9,
                smd.nu : 0.25,
                smd.rho: 1000,
                smd.pstress: True})
    a0 = 1/(1e4*np.pi)
    tau8 = 1e6
    K2sol = 1e4
    K1sol = 0.
    
    K1st2 = K1_static(a0,'bilat_bimat_inf_domain',
                      matprop1=mat1, matprop2=mat1,
                      tauinf=tau8)

    if abs(K1st2 - K1sol) > 1e-6*K1sol:
        print(K1st2)
        raise RuntimeError('Failed pseudo bimat K1')
        
    K2st2 = K2_static(a0,'bilat_bimat_inf_domain',
                      matprop1=mat1, matprop2=mat1,
                      tauinf=tau8)

    if abs(K2st2 - K2sol) > 1e-6*K2sol:
        print(K2st2)
        raise RuntimeError('Failed pseudo bimat K2')


# ----------------------------------------------
if (__name__ == '__main__'):
    print('unit tests: lefm stress intensity factor (sif)')
    test_sif_unimaterial()
    test_sif_bimaterial_1()
    print('success!')
