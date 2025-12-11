import sys
if not '-m' in sys.argv:
    from .functions import *
    from .fields import *
    from .energies import *
    from .sif import *
    from .dynamics import *
    from .cohesive_zone import *
