#!/bin/bash

(cd .. && python3 -m solidmechanics.utilities)
(cd .. && python3 -m solidmechanics.symmetrictensor)
(cd .. && python3 -m solidmechanics.cauchystress)
(cd .. && python3 -m solidmechanics.infinitesimalstrain)
(cd .. && python3 -m solidmechanics.constitutive_law)
(cd .. && python3 -m solidmechanics.linearelasticity)

(cd .. && python3 -m solidmechanics.lefm.utilities)
(cd .. && python3 -m solidmechanics.lefm.fields)
(cd .. && python3 -m solidmechanics.lefm.sif)
(cd .. && python3 -m solidmechanics.lefm.cohesive_zone)
(cd .. && python3 -m solidmechanics.lefm.dynamics)
(cd .. && python3 -m solidmechanics.lefm.energies)

