###############################################################################
# packages #
###############################################################################
import sys
import numpy as np
import time
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

###############################################################################
# my functions #
###############################################################################
# function that determines Nup from s,N,Sz
## GC, 23-05-2022
def calc_Nup(s,N,Sz):
	'''
		function that determines Nup (see QuSpin documentation) from s,N,Sz
	'''

	if (2*s)%2 == 0 and (2*Sz)%2 == 0:
		Nup = int(s*N+Sz)
	elif (2*s)%2 == 1 and abs(2*Sz)%2 == N%2:
		Nup = int(s*N+Sz)
	else:
		print('ERROR: check s, N and Sz')
		sys.exit()

	return Nup

# S^2 operator
## GC, 11-01-2022
def S2_op(N,basis):
	'''
		S^2 operator

		S^2 = SpSm + Sz^2 - Sz
		SpSm = sum_{i,j} Sp(i)Sm(j)
	'''

	Sz = Sz_op(N,basis)

	# S+S-
	SpSm = [[1,i,j] for i in range(N) for j in range(N)]

	static = [
		['+-',SpSm]
	]
	dynamic = []

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	SpSm = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)

	S2 = SpSm + np.dot(Sz,Sz) - Sz

	return S2

# Sz operator
## GC, 11-01-2022
def Sz_op(N,basis):
	'''
		Sz operator

		Sz = sum_i Sz(i)
	'''

	Sz = Szi_op(0,basis)
	for i in range(1,N):
		Sz += Szi_op(i,basis)

	return Sz

# Sz(i) operator
## GC, 11-01-2022
def Szi_op(i,basis):
	'''
		Sz(i) operator
	'''

	Sz = [[1,i]]

	static = [
		['z',Sz]
	]
	dynamic = []

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	Szi = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)

	return Szi

# J2-J1 Heisenberg chain
## GC, 07-12-2022
def H_Heis_J2J1_1D(N,J2,J1,basis,BC='OBC'):
	'''
		J2-J1 Heisenberg chain

		(0)--(1)...(2)

		-- , J2
		... , J1
	'''

	SzSz = []
	SpSm  = []
	SmSp = []

	# J2
	if BC=='OBC':
		SzSz += [[J2,i,i+1] for i in range(0,N-1,2)]
		SpSm += [[J2/2,i,i+1] for i in range(0,N-1,2)]
		SmSp += [[J2/2,i,i+1] for i in range(0,N-1,2)]
	elif BC=='PBC':
		SzSz += [[J2,i,(i+1)%N] for i in range(0,N,2)]
		SpSm += [[J2/2,i,(i+1)%N] for i in range(0,N,2)]
		SmSp += [[J2/2,i,(i+1)%N] for i in range(0,N,2)]

	# J1
	if BC=='OBC':
		SzSz += [[J1,i,i+1] for i in range(1,N-1,2)]
		SpSm += [[J1/2,i,i+1] for i in range(1,N-1,2)]
		SmSp += [[J1/2,i,i+1] for i in range(1,N-1,2)]
	elif BC=='PBC':
		SzSz += [[J1,i,(i+1)%N] for i in range(1,N,2)]
		SpSm += [[J1/2,i,(i+1)%N] for i in range(1,N,2)]
		SmSp += [[J1/2,i,(i+1)%N] for i in range(1,N,2)]

	static = [
		["zz",SzSz],
		["+-",SpSm],
		["-+",SmSp]
	]
	dynamic = []

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	H = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)

	return H

###############################################################################
# main #
###############################################################################
start_time = time.time()

# physical parameters
s = 1/2
N = 10
J2 = 1
J1 = J2

# other parameters
Sz = 0
nexc = 3

# open files
fw = open("E-S2.txt", "w")

# basis
Nup = calc_Nup(s,N,Sz)
if (2*s)%2 == 0:
	basis = spin_basis_1d(N, Nup=Nup, S=str(int(s)), pauli=False)
else:
	basis = spin_basis_1d(N, Nup=Nup, S=str(int(2*s)) + '/2', pauli=False)

# Hamiltonian
H = H_Heis_J2J1_1D(N,J2,J1,basis)

# diagonalization
if N==2:
    En,psin = H.eigh()
else:
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

# close files
fw.close()
###############################################################################
