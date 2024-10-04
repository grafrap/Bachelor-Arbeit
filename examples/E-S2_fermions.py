###############################################################################
# packages #
###############################################################################
import sys
import numpy as np
import time
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian

###############################################################################
# my functions #
###############################################################################
# Hubbard chain
## GC, 14-01-2022
def H_Hubb_1D(Nsites,t,U,basis,BC='OBC'):
	'''
		Hubbard chain

		(0)---(1)

	    --- , -t
	'''

	CdagsCs = [] #s=up,dn
	Nupdn = []

	# t
	if BC=='OBC':
		CdagsCs += [[-t,i,i+1] for i in range(Nsites-1)]
		#hc
		CdagsCs += [[-t,i+1,i] for i in range(Nsites-1)]
	elif BC=='PBC':
		CdagsCs += [[-t,i,(i+1)%Nsites] for i in range(Nsites)]
		#hc
		CdagsCs += [[-t,(i+1)%Nsites,i] for i in range(Nsites)]

	# U
	Nupdn += [[U,i,i] for i in range(Nsites)]

	static = [
		['+-|',CdagsCs], #CdagupCup
		['|+-',CdagsCs], #CdagdnCdn
		['n|n',Nupdn]
	]
	dynamic = []

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	H = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)

	return H

# Sz(i) operator
## GC, 30-11-2021
def Szi_op(i,basis):
	'''
		Sz(i) operator

		Sz(i) = 1/2*[ Nup(i)-Ndn(i) ]
	'''

	Nup = [[0.5,i]]
	Ndn = [[-0.5,i]]

	static = [
		['n|',Nup],
		['|n',Ndn]
	]
	dynamic = []

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	Szi = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)

	return Szi

# Sz operator
## GC, 30-11-2021
def Sz_op(Nsites,basis):
	'''
		Sz operator

		Sz = sum_i Sz(i)
	'''

	Sz = Szi_op(0,basis)
	for i in range(1,Nsites):
		Sz += Szi_op(i,basis)

	return Sz

# S^2 operator
## GC, 30-11-2021
def S2_op(Nsites,basis):
	'''
		S^2 operator

		S^2 = SpSm + Sz^2 - Sz
		SpSm = sum_{i,j} Sp(i)Sm(j) = sum_{i,j} Cdagup(i)Cdn(i)Cdagdn(j)Cup(j)
	'''

	CdagupCdnCdagdnCup = [[1.0,i,j,i,j] for i in range(Nsites)
		for j in range(Nsites)]

	static = [
		['+-|-+',CdagupCdnCdagdnCup]
	]
	dynamic = []

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	SpSm = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)

	Sz = Sz_op(Nsites,basis)
	S2 = SpSm + np.dot(Sz,Sz) - Sz

	return S2

# function that determines Ne_up,Ne_dn from Ne,Sz
## GC, 18-01-2022
def calc_Neupdn(Ne,Sz):
	'''
		function that determines Ne_up,Ne_dn from Ne,Sz
	'''

	Ne_up,Ne_dn = None,None
	if Ne%2 == 0 and (2*Sz)%2 == 0:
		Ne_up = Ne//2 + Sz
		Ne_dn = Ne//2 - Sz
	elif Ne%2 == 1 and abs(2*Sz)%2 == 1:
		Ne_up = Ne//2 + (Sz+1/2)
		Ne_dn = Ne//2 - (Sz-1/2)
	else:
		print('ERROR: check Ne and Sz')
		sys.exit()

	return int(Ne_up),int(Ne_dn)

###############################################################################
# main #
###############################################################################
start_time = time.time()

# physical parameters
N = 8
t = 1
U = 4*t

# other parameters
Ne = N
Sz = 0
nexc = 3

# open writing file
fw = open("E-S2_fermions.txt", "w")

# Ne_up, Ne_dn
Ne_up,Ne_dn = calc_Neupdn(Ne,Sz)

# basis
basis = spinful_fermion_basis_1d(N,Nf=(Ne_up,Ne_dn))

# Hamiltonian
H = H_Hubb_1D(N,t,U,basis)

# diagonalization
En,psin = H.eigsh(k=nexc+1, which='SA')

# S^2 operator
S2op = S2_op(N,basis)

# <psin|S^2|psin>
S2n = [np.dot(psin[:,n].conj(),S2op.dot(psin[:,n])) for n in range(nexc+1)]

# outputs
fw.write("#List of E:\n")
fw.write(np.array2string(En, separator=', ') + "\n")
fw.write("\n")
fw.write("#List of S^2:\n")
fw.write(str(S2n) + "\n")
fw.write("--------------------\n\n")
fw.write("total time = " + str(time.time() - start_time) + " s\n")

## close file
fw.close()
###############################################################################
