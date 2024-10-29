###############################################################################
# packages #
###############################################################################
import sys
import numpy as np
import time
from quspin.basis import spin_basis_1d
'''
	Note: with 'spin_basis_general', we can use 'Op_shift_sector' to compute 
		operators between states in different symmetry sectors
'''
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
		SzSz += [[J2,i,i+2] for i in range(0,N-2,1)]
		SpSm += [[J2/2,i,i+2] for i in range(0,N-2,1)]
		SmSp += [[J2/2,i,i+2] for i in range(0,N-2,1)]
	elif BC=='PBC':
		SzSz += [[J2,i,(i+2)%N] for i in range(0,N,1)]
		SpSm += [[J2/2,i,(i+2)%N] for i in range(0,N,1)]
		SmSp += [[J2/2,i,(i+2)%N] for i in range(0,N,1)]

	# J1
	if BC=='OBC':
		SzSz += [[J1 + (-1)**(i+1) * 0.03 * J1,i,i+1] for i in range(0,N-1,1)]
		SpSm += [[J1/2 + (-1)**(i+1) * 0.03 * J1 / 2,i,i+1] for i in range(0,N-1,1)]
		SmSp += [[J1/2 + (-1)**(i+1) * 0.03 * J1 / 2,i,i+1] for i in range(0,N-1,1)]
	elif BC=='PBC':
		SzSz += [[J1 + (-1)**i * 0.03 * J1 / 2,i,(i+1)%N] for i in range(0,N,1)]
		SpSm += [[J1/2 + (-1)**i * 0.03 * J1 / 2,i,(i+1)%N] for i in range(0,N,1)]
		SmSp += [[J1/2 + (-1)**i * 0.03 * J1 / 2,i,(i+1)%N] for i in range(0,N,1)]

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
'''
	Hubbard model at half filling with U >> t maps to spin-1/2 Heisenberg
		model with J = 4*t^2/U
'''
s = 1.5
N = int(sys.argv[1])
J = 10

# other parameters
'''
	Sz symmetry sector that includes all states:
		- if s is integer, Sz=0
		- if s is half integer, Sz=N%2
'''
if (2*s)%2==0 or N%2==0: 
	Sz = 0
else:
	Sz = 1/2
nexc = 0
Sz = 1/2

# open files
fw = open("N_" + str(N) + ".txt", "w")

# basis
Nup = calc_Nup(s,N,Sz)
if (2*s)%2 == 0:
	basis = spin_basis_1d(N, Nup=Nup, S=str(int(s)), pauli=False)
else:
	basis = spin_basis_1d(N, Nup=Nup, S=str(int(2*s)) + '/2', pauli=False)

# Hamiltonian
H = H_Heis_J2J1_1D(N,0.19*J,J,basis)

# diagonalization
if nexc+1 < basis.Ns/10: #use Lanczos when states << Hilbert space
    En,psin = H.eigsh(k=nexc+1, which='SA')
else:
	En,psin = H.eigh()

H_max = -H

if nexc+1 < basis.Ns/10: #use Lanczos when states << Hilbert space
  En_max,psin_max = H_max.eigsh(k=nexc+1, which='SA')
else:
	En_max,psin_max = H_max.eigh()

# Sz(i) operator
Sziop = [Szi_op(i,basis) for i in range(N)]

# <psin|Sz(i)|psin>
Szin = [[np.dot(psin[:,n].conj(),Sziop[i].dot(psin[:,n])) for i in range(N)]
	for n in range(len(En))]

# outputs
fw.write("#List of E:\n")
fw.write(np.array2string(En, separator=', ') + np.array2string(En_max, separator=', ') + "\n")
fw.write("\n")
fw.write("#List of Sz(i):\n")
fw.write(str(Szin) + "\n")
fw.write("--------------------\n\n")
fw.write("total time = " + str(time.time() - start_time) + " s\n")

# close file
fw.close()
