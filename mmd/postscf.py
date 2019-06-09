from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import itertools

class PostSCF(object):
    """Class for post-scf routines"""
    def __init__(self,mol):
        self.mol = mol
        if not self.mol.is_converged:
            sys.exit("SCF not converged, skipping Post-SCF")
        self.spin_block()
        self.ao2mo()

    def spin_block(self):
        # These are all done in the MO basis
        self.mol.spin_orbital = True
        # Core Hamiltonian first
        self.mol.F = np.kron(self.mol.F,np.eye(2))
        # TEI
        self.mol.double_bar = np.kron(np.kron(self.mol.TwoE,np.eye(2)).T,np.eye(2)) 
        # Physicist notation and antisymmetrize
        self.mol.double_bar = self.mol.double_bar.transpose(0,2,1,3) \
                            - self.mol.double_bar.transpose(0,2,3,1)
        # Spin block MO coefficients, too
        self.mol.C = np.kron(self.mol.C,np.eye(2))
        self.mol.nocc *= 2
        self.mol.nbasis *= 2

    def ao2mo(self):
        """Routine to convert AO integrals to MO integrals"""
        if self.mol.spin_orbital:
           # H is MO core hamiltonian, G is the MO basis TEI
            self.mol.H = np.einsum('pQ, pP -> PQ', 
                         np.einsum('pq, qQ -> pQ', self.mol.F, self.mol.C), 
                                   self.mol.C)
            self.mol.G = np.einsum('pQRS, pP -> PQRS',
                         np.einsum('pqRS, qQ -> pQRS',
                         np.einsum('pqrS, rR -> pqRS', 
                         np.einsum('pqrs, sS -> pqrS', self.mol.double_bar, self.mol.C), 
                                   self.mol.C), self.mol.C), self.mol.C)
        else:
            self.mol.single_bar = np.einsum('mp,mnlz->pnlz',
                                            self.mol.C,self.mol.TwoE)
            temp = np.einsum('nq,pnlz->pqlz',
                             self.mol.C,self.mol.single_bar)
            self.mol.single_bar = np.einsum('lr,pqlz->pqrz',
                                            self.mol.C,temp)
            temp = np.einsum('zs,pqrz->pqrs',
                             self.mol.C,self.mol.single_bar)
            self.mol.single_bar = temp
    
    def MP2(self):
        """Routine to compute MP2 energy from RHF reference"""

        EMP2 = 0.0
        occupied = range(self.mol.nocc)
        virtual  = range(self.mol.nocc,self.mol.nbasis)

        if self.mol.spin_orbital:
            MO = np.diagonal(self.mol.H)
            for i,j,a,b in itertools.product(occupied,occupied,virtual,virtual): 
                denom = MO[i] + MO[j] - MO[a] - MO[b] 
                numer = self.mol.G[i,j,a,b]*self.mol.G[i,j,a,b] 
                EMP2 += numer/denom
            EMP2 *= 0.25
        else:
            for i,j,a,b in itertools.product(occupied,occupied,virtual,virtual): 
                denom = self.mol.MO[i] + self.mol.MO[j] \
                      - self.mol.MO[a] - self.mol.MO[b]
                numer = self.mol.single_bar[i,a,j,b] \
                      * (2.0*self.mol.single_bar[i,a,j,b] 
                        - self.mol.single_bar[i,b,j,a])
                EMP2 += numer/denom
        self.mol.emp2 = EMP2 + self.mol.energy   
        print('E(MP2) = ', self.mol.emp2.real) 

