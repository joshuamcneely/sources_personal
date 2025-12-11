# linearelasticity.py
#
# functions from linear elasticity
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @author Gabriele Albertini <ga288@cornell.edu>
# @date     2015/10/01
# @modified 2020/12/12

from __future__ import print_function, division, absolute_import

import numpy as np
import math
from scipy.optimize import bisect

from .definitions import smd
from .cauchystress import CauchyStress
from .infinitesimalstrain import InfinitesimalStrain
from .constitutive_law import ConstitutiveLaw

class LinearElasticMaterial(ConstitutiveLaw):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complete()
        self.complete_wave_speeds()
    
    def complete(self):
        """Complete a set of material properties with mu, lambda and kpa."""
        
        if smd.E in self and smd.nu in self:
            E  = float(self[smd.E])
            nu = float(self[smd.nu])

            self[smd.mu] = E / (2. * (1 + nu))

            if self.is_plane_strain():
                self[smd.llambda] = nu * E / ((1. + nu) * (1. - 2. * nu))
            else:
                self[smd.llambda] = nu * E / ((1. + nu)*(1. - nu))

            self[smd.kpa] = self[smd.llambda] + 2./3. * self[smd.mu]

        elif smd.llambda in self and smd.mu in self:
            llambda = float(self[smd.llambda])
            mu      = float(self[smd.mu])

            self[smd.nu] = llambda / (2*(llambda + mu))
            self[smd.E]  = mu*(3*llambda + 2*mu)/(llambda +mu)
        elif smd.cp in self and smd.cs in self and smd.rho in self:
            rho = self[smd.rho]
            cs = self[smd.cs]
            cp = self[smd.cp]

            mu = rho*cs**2
            llambda = rho*cp**2 - 2*mu

            self[smd.mu]=mu
            self[smd.llambda]=llambda
            self.complete()
        else:
            import warnings
            warnings.warn("material not completed (missing properties)!")

    def complete_wave_speeds(self):
        """Complete the material properties with 
        the P-wave, S-wave and Rayleigh wave speed
        """
        if smd.rho in self:
            self.complete_p_wave()
            self.complete_s_wave()
            self.complete_R_wave()
            self[smd.k]=self[smd.cs]/self[smd.cp]

    def complete_p_wave(self):
        """Complete the material properties with the P-wave speed."""

        E   = float(self[smd.E])
        nu  = float(self[smd.nu])
        rho = float(self[smd.rho])

        # P-wave
        if self.is_plane_strain():
            pwave = math.sqrt(E * ( 1 - nu ) / ((1+nu)*(1 - 2*nu)) / rho)
        else:
            pwave = math.sqrt(E / (1 - nu**2) / rho)
        self[smd.cp] = pwave


    def complete_s_wave(self):
        """Complete the material properties with the S-wave speed."""

        E   = float(self[smd.E])
        nu  = float(self[smd.nu])
        rho = float(self[smd.rho])

        # S-wave
        swave = math.sqrt(E / (2 * rho * (1 + nu)))
        self[smd.cs] = swave

        
    def complete_R_wave(self):
        """Complete the material properties with the Rayleigh-wave speed."""

        nu    = float(self[smd.nu])
        pwave = float(self[smd.cp])
        swave = float(self[smd.cs])

        # R-wave estimate (works well for plane strain but not so much for plane stress)
        Rwave_est = (0.87 + 1.12*nu) / (1 + nu) * swave 

        # D function as function to find root
        def find_cR(v,*args):
            """Compute D as defined in Freund p.162 eq 4.3.8."""
            c_s = args[0][0]
            c_d = args[0][1]
            alpha_s = math.sqrt(1 - v**2/c_s**2)
            alpha_d = math.sqrt(1 - v**2/c_d**2)
            return 4*alpha_d*alpha_s - (1 + alpha_s**2)**2
        
        # Freund p.162 eq 4.3.8: D=0 for v=cR
        cRmin = 0.2*swave  # according to estimate: cR = 0.32*cs for nu = 0.5
        cRmax = 0.95*swave # according to estimate: cR = 0.87*cs for nu = 0
        Rwave = bisect(find_cR,cRmin,cRmax,args=[swave,pwave])

        self[smd.cR] = Rwave




    # ------------------------------------------------------------------------------
    def stresses_to_strains(self,**kwargs):

        sig11 = kwargs.get('sig11')
        sig22 = kwargs.get('sig22')
        sig12 = kwargs.get('sig12')

        # 2D
        if 'sig33' not in kwargs:
            return self.stresses_to_strains_2d(sig11,
                                               sig22,
                                               sig12,
                                               **kwargs)
        else:
            sig33 = kwargs.get('sig33')
            sig13 = kwargs.get('sig13')
            sig23 = kwargs.get('sig23')
            return self.stresses_to_strains_2d(sig11,
                                               sig22,
                                               sig33,
                                               sig12,
                                               sig13,
                                               sig23,
                                               **kwargs)


    def stresses_to_strains_2d(self,sig11,sig22,sig12,**kwargs):

        # check if it is a 2D numpy array and if not make it one
        try:
            sig11.shape[1]
        except:
            np1D = True
            sig11 = np.array([sig11])
            sig22 = np.array([sig22])
            sig12 = np.array([sig12])

        eps11 = np.zeros_like(sig11)
        eps22 = np.zeros_like(sig22)
        eps12 = np.zeros_like(sig12)

        stress = CauchyStress()

        for (i,j),tmp in np.ndenumerate(sig11):
            stress[0,0] = sig11[i,j]
            stress[1,1] = sig22[i,j]
            stress[0,1] = sig12[i,j]
            stress[1,0] = stress[0,1]

            strain = self.stress_to_strain_2d(stress)

            eps11[i,j] = strain[0,0]
            eps22[i,j] = strain[1,1]
            eps12[i,j] = strain[1,0]

        if np1D:
            eps11 = eps11[0]
            eps22 = eps22[0]
            eps12 = eps12[0]

        return eps11,eps22,eps12


    def stresses_to_strains_3d(self,sig11,sig22,sig33,sig12,sig13,sig23,**kwargs):

        # check if it is a 2D numpy array and if not make it one
        try:
            sig11.shape[1]
        except:
            np1D = True
            sig11 = np.array([sig11])
            sig22 = np.array([sig22])
            sig33 = np.array([sig33])
            sig12 = np.array([sig12])
            sig13 = np.array([sig13])
            sig23 = np.array([sig23])

        eps11 = np.zeros_like(sig11)
        eps22 = np.zeros_like(sig22)
        eps33 = np.zeros_like(sig33)
        eps12 = np.zeros_like(sig12)
        eps13 = np.zeros_like(sig13)
        eps23 = np.zeros_like(sig23)

        stress = CauchyStress()

        for (i,j),tmp in np.ndenumerate(sig11):
            stress[0,0] = sig11[i,j]
            stress[1,1] = sig22[i,j]
            stress[2,2] = sig33[i,j]
            stress[0,1] = sig12[i,j]
            stress[1,0] = stress[0,1]
            stress[0,2] = sig13[i,j]
            stress[2,0] = stress[0,2]
            stress[1,2] = sig23[i,j]
            stress[2,1] = stress[1,2]

            strain = self.stress_to_strain_3d(stress)

            eps11[i,j] = strain[0,0]
            eps22[i,j] = strain[1,1]
            eps33[i,j] = strain[2,2]
            eps12[i,j] = strain[0,1]
            eps13[i,j] = strain[0,2]
            eps23[i,j] = strain[1,2]

        if np1D:
            eps11 = eps11[0]
            eps22 = eps22[0]
            eps33 = eps33[0]
            eps12 = eps12[0]
            eps13 = eps13[0]
            eps23 = eps23[0]

        return eps11,eps22,eps33,eps12,eps13,eps23



    def stress_to_strain(self,stress):
        """ Compute strain based on stress and material properties

        Args:
           stress(list)

        Return:
           InfinitesimalStrain object
    """
        if stress.dim == 2:
            return self.stress_to_strain_2d(stress)
        elif stress.dim == 3:
            return self.stress_to_strain_3d(stress)

    def stress_to_strain_2d(self,stress):
        """Compute strain based on stress and material properties for 2d case

        Args:
           stress(CauchyStress)

        Return:
           InfinitesimalStrain object
        """
        strain = InfinitesimalStrain()

        E  = float(self[smd.E])
        nu = float(self[smd.nu])

        if self.is_plane_strain():
            strain[0,0] = (1. + nu) / E * ((1-nu) * stress[0,0] - nu * stress[1,1])
            strain[1,1] = (1. + nu) / E * ((1-nu) * stress[1,1] - nu * stress[0,0])
        else:
            strain[0,0] = 1. / E * (stress[0,0] - nu * stress[1,1])
            strain[1,1] = 1. / E * (stress[1,1] - nu * stress[0,0])

        strain[0,1] = (1. + nu) / E * stress[0,1]
        strain[1,0] = strain[0,1]

        return strain

    def stress_to_strain_3d(self,stress):

        #strain = InfinitesimalStrain()

        E  = float(self[smd.E])
        nu = float(self[smd.nu])

        strain = 1.0 / E * ((1 + nu) * stress - nu * np.identity(3)*np.trace(stress))

        return InfinitesimalStrain(strain)

    # ------------------------------------------------------------------------------
    def strain_to_stress(self,strain):
        """ Compute stress based on strain and material properties

        Note: This function is not yet implemented for 3D

        Args:
           strain(list)

        Return:
           CauchyStress object

        """
        if strain.dim == 2:
            return self.strain_to_stress_2d(strain)
        elif strain.dim == 3:
            return self.strain_to_stress_3d(strain)

    def strain_to_stress_2d(self,strain,stress=None):
        """2d case
        """
        if stress==None:
            stress = CauchyStress(np.zeros(strain.shape))

        E  = float(self[smd.E])
        nu = float(self[smd.nu])

        if self.is_plane_strain():
            stress[0,0] = E / (1. + nu) / (1. - 2*nu) * ((1-nu) * strain[0,0] + nu * strain[1,1])
            stress[1,1] = E / (1. + nu) / (1. - 2*nu) * ((1-nu) * strain[1,1] + nu * strain[0,0])
        else:
            stress[0,0] = E / (1. - nu*nu) * (strain[0,0] + nu * strain[1,1])
            stress[1,1] = E / (1. - nu*nu) * (strain[1,1] + nu * strain[0,0])

        stress[0,1] = E / (1. + nu) * 0.5*(strain[0,1] + strain[1,0])
        stress[1,0] = stress[0,1]

        return stress

    def strain_to_stress_3d(self,strain):
        """ 3d case
        """
        ll = self[smd.llambda]
        mu = self[smd.mu]
        stress = ll*np.identity(3)*np.trace(strain) + 2*mu*strain
        return CauchyStress(stress)

    # -----------------------------------------------------------------------------
    def strain_energy_density(self,strain,**kwargs):
        """ Compute strain energy density

        Args:
           strain(InfinitesimalStrain)

        Return:
           strain energy density(float)

        """
        stress = self.strain_to_stress(strain)

        return float(0.5*np.sum(stress*strain))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def RT_potential(p, matprop1, matprop2):
    """Compute reflection and transmission coefficient for pwave

    Based on Aki and Richards 1980

    Args:
       p(): sin(i)/cp ray parameter = horizontal slowness \
            (a scalar or np.array row vector 
            (dtype=np.complex) may be useful)
       matprop1(): material properties for incident wave
       matprop2(): material properties for transmitted wave

    Return:
       a1, b1, r1 (float): reflection coefficient for pwave
       a2, b2, r2 (float): transmission coefficient for pwave
    """ 
    print('incident wave model')
    complete(matprop1)
    complete_wave_speeds(matprop1)
    print('reflected wave model')
    complete(matprop2)
    complete_wave_speeds(matprop2)
    
    a1, b1, r1 = [matprop1[smd.cp],matprop1[smd.cs],matprop1[smd.rho]]
    a2, b2, r2 = [matprop2[smd.cp],matprop2[smd.cs],matprop2[smd.rho]]
    #work in progress
    return RuntimeError()
    


def RT_displacement_field(p, matprop1, matprop2):
    
    """Compute reflection and transmission coefficient of displacement fields

    Based on Aki and Richards 1980 eq(5.38)

    Args:
       p(): sin(i)/cp ray parameter = horizontal slowness \
            (a scalar or np.array row vector 
            (dtype=np.complex) may be useful)
       matprop1(): material properties for incident wave
       matprop2(): material properties for transmitted wave

    Return:
       RTmatrix(numpy.array): contains the coefficient for reflection and transmission of displacement fields.
    """
    
    p = np.array(p)
    try:
        a1, b1, r1 = [matprop1[smd.cp],matprop1[smd.cs],matprop1[smd.rho]]
        a2, b2, r2 = [matprop2[smd.cp],matprop2[smd.cs],matprop2[smd.rho]]
    except:
        print('incident wave model')
        complete(matprop1)
        complete_wave_speeds(matprop1)
        print('reflected wave model')
        complete(matprop2)
        complete_wave_speeds(matprop2)
        a1, b1, r1 = [matprop1[smd.cp],matprop1[smd.cs],matprop1[smd.rho]]
        a2, b2, r2 = [matprop2[smd.cp],matprop2[smd.cs],matprop2[smd.rho]]
        
    # vertical slownesses
    etaa1 = np.sqrt(1./(a1**2.) - p*p)
    etaa2 = np.sqrt(1./(a2**2.) - p*p)
    etab1 = np.sqrt(1./(b1**2.) - p*p)
    etab2 = np.sqrt(1./(b2**2.) - p*p)

    a = r2*(1.-2.*b2**2.*p*p)-r1*(1.-2.*b1**2.*p*p) 
    b = r2*(1.-2.*b2**2.*p*p)+2.*r1*b1**2.*p*p 
    c = r1*(1.-2.*b1**2.*p*p)+2.*r2*b2**2.*p*p 
    d = 2.*(r2*b2**2.-r1*b1**2.)

    E = b * etaa1 + c * etaa2 
    F = b * etab1 + c * etab2 
    G = a - d * etaa1 * etab2 
    H = a - d * etaa2 * etab1 
    D = E*F + G*H*p*p 

    detM = D * a1 *a2 * b1 * b2
    
    Rpp = ((b*etaa1-c*etaa2)*F - (a + d*etaa1*etab2)*H *p *p) /D 
    Rps = -(2.  * etaa1  * (a  * b + d * c  * etaa2  * etab2)  * p * a1/b1 ) /D 
    Rss = -((b *etab1-c *etab2) *E-(a+d *etaa2 *etab1) *G *p *p) /D 
    Rsp = -(2. *etab1 *(a *b+d*c *etaa2 *etab2) *p*(b1/a1)) /D 
    Tpp = (2. *r1*etaa1 *F*(a1/a2)) /D 
    Tps = (2. *r1*etaa1 *H *p*(a1/b2)) /D 
    Tss = 2. *r1*etab1 *E*(b1/b2) /D 
    Tsp = -2.*(r1*etab1 *G *p*(b1/a2)) /D 

    RTmatrix=np.zeros((4,2,len(p)))    

    for n in range(len(p)):
        RTmatrix[:,:,n]=[[Rpp[n],Rsp[n]],
                         [Rps[n],Rss[n]],
                         [Tpp[n],Tsp[n]],
                         [Tps[n],Tss[n]]]

    return RTmatrix

# ---------------------------------------------------------
def test_linearelasticity():

    # test for plane stress
    mat1 = LinearElasticMaterial({smd.E : 5.65e9,
                                  smd.nu : 0.33,
                                  smd.rho : 1180,
                                  smd.pstress : True})

    cs1 = CauchyStress([[3,2],
                        [2,3]])

    # compute strain from stress and then inverse and verify
    is1 = mat1.stress_to_strain(cs1)
    cs2 = mat1.strain_to_stress(is1)
    if cs1 != cs2:
        print(cs1)
        print(is1)
        print(cs2)
        raise RuntimeError('stress-strain relation for plane stress wrong')

    # test for plane strain
    mat2 = LinearElasticMaterial({smd.E : 5.65e9,
                                  smd.nu : 0.33,
                                  smd.rho : 1180,
                                  smd.pstress : False})
    
    # compute strain from stress and then inverse and verify
    is1 = mat2.stress_to_strain(cs1)
    cs2 = mat2.strain_to_stress(is1)
    if cs1 != cs2:
        print(cs1)
        print(is1)
        print(cs2)
        raise RuntimeError('stress-strain relation for plane strain wrong')

    
# ---------------------------------------------------------
if (__name__ == '__main__'):
    print('unit tests: linear elasticity')
    test_linearelasticity()
    print('success!')
