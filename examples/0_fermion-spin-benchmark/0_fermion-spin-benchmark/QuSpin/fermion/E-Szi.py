###############################################################################
# packages #
###############################################################################
import sys
import numpy as np
import time
from quspin.basis import spinful_fermion_basis_1d
'''
	Note: with 'spinful_fermion_basis_general', we can use 'Op_shift_sector' 
		to compute operators between states in different symmetry sectors
'''
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
N = int(sys.argv[1])
t = 1
U = 10*t

# other parameters
Ne = N #half-filling
Sz = (Ne%2)/2 #Sz symmetry sector that includes all states
nexc = 1

# open writing file
fw = open("N_" + str(N) + ".txt", "w")

# Ne_up, Ne_dn
Ne_up,Ne_dn = calc_Neupdn(Ne,Sz)

# basis
basis = spinful_fermion_basis_1d(N,Nf=(Ne_up,Ne_dn))

# Hamiltonian
H = H_Hubb_1D(N,t,U,basis)

# diagonalization
if nexc+1 < basis.Ns/10: #use Lanczos when states << Hilbert space
    En,psin = H.eigsh(k=nexc+1, which='SA')
else:
	En,psin = H.eigh()

# Sz(i) operator
Sziop = [Szi_op(i,basis) for i in range(N)]

# <psin|Sz(i)|psin>
Szin = [[np.dot(psin[:,n].conj(),Sziop[i].dot(psin[:,n])) for i in range(N)]
	for n in range(len(En))]

# outputs
fw.write("#List of E:\n")
fw.write(np.array2string(En, separator=', ') + "\n")
fw.write("\n")
fw.write("#List of Sz(i):\n")
fw.write(str(Szin) + "\n")
fw.write("--------------------\n\n")
fw.write("total time = " + str(time.time() - start_time) + " s\n")

# close file
fw.close()