import sys
if not '-m' in sys.argv:
    from .utilities import *
    from .definitions import smd
    from .cauchystress import CauchyStress
    from .infinitesimalstrain import InfinitesimalStrain
    from .constitutive_law import ConstitutiveLaw
    from .linearelasticity import LinearElasticMaterial
