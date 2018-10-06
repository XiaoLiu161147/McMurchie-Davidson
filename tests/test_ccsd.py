import unittest
import numpy as np
from mmd.integrals import *
from mmd.molecule import *
from mmd.scf import *
from mmd.postscf import *

water = """
0 1
O  0.000000000000  -0.0757918436   0.000000000000
H  0.866811829   0.6014357793  -0.000000000000
H -0.866811829   0.6014357793  -0.000000000000
"""

heh = """
1 1
H 0.0 0.0 0.0
He 0.0 0.0 0.9295
"""


class test_CCSD(unittest.TestCase):
    def test_heh_sto3g(self):
        mol = Molecule(geometry=heh,basis='sto-3g')
        mol.RHF(direct=False) # we need the two-electron integrals
        PostSCF(mol).CCSD()
        self.assertAlmostEqual(mol.energy.real,-2.854368651625)
        self.assertAlmostEqual(mol.eccsd.real,-2.862594375658)

    def test_water_sto3g(self):
        mol = Molecule(geometry=water,basis='sto-3g')
        mol.RHF(direct=False) # we need the two-electron integrals
        PostSCF(mol).CCSD()
        self.assertAlmostEqual(mol.energy.real,-74.942079928101)
        self.assertAlmostEqual(mol.eccsd.real,-75.01276001566)


