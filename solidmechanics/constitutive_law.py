# constitutive_law.py
#
# Constitutive Material law
# 
# There is no warranty for this code
#
# @version 1.0
# @author David Kammer <dkammer@ethz.ch>
# @date     2020/12/13
# @modified 2020/12/13

from __future__ import print_function, division, absolute_import

from .definitions import smd

class ConstitutiveLaw(dict):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def is_plane_stress(self):
        """Check if the material is in plane stress conditions

        Return
           bool: True if succesful"""

        if smd.pstress in self and smd.pstrain in self:
            if self[smd.pstress] and not self[smd.pstrain]:
                return True
            elif not self[smd.pstress] and self[smd.pstrain]:
                return False
            else:
                print('pstress=',self[smd.pstress])
                print('pstrain=',self[smd.pstrain])
                raise('Plane stress not equal plane strain information')

        elif smd.pstress in self:
            return self[smd.pstress]
        elif smd.pstrain in self:
            return not self[smd.pstrain]
        else:
            return False # it is a 3D material

    def is_plane_strain(self):
        """Check if the material is in plane strain conditions

        Return
           bool: True if succesful"""

        return not self.is_plane_stress()


def test_constitutivelaw():
    pass

# ---------------------------------------------------------
if (__name__ == '__main__'):
    print('unit tests: constitutive law')
    test_constitutivelaw()
    print('nothing done')
