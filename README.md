# First-Repository
Repository with ODE/PDE solutions.

"""
Solution to system of two PDEs about the model bidomain model to application is in cardiac electrophysiology.

Taken from the article: 
Using Python to Solve Partial Differential Equations. 
Mardal et al., 2007
http://csc.ucdavis.edu/~cmg/Group/readings/pythonissue_3of4.pdf
Acess in 04/05/2016
"""

from dolfin import Mesh
from pycc.MatSparse import *
import numpy
from pycc import MatFac
from pycc import ConjGrad
from pycc.BlockMatrix import *
from pycc.Functions import *
from pycc.ODESystem import *
from pycc.CondGen import *
from pycc.IonicODEs import *

mesh = Mesh("Heart.xml.gz")
matfac = MatFac.MatrixFactory(mesh)
M = matfac.computeMassMatrix()
pc = PyCond("Heart.axis")
pc.setconductances(3.0e-3, 3e-4)
ct = ConductivityTensorFunction(
pc.conductivity)
Ai = matfac.computeStiffnessMatrix(ct)
pc.setconductances(5.0e-3, 1.6e-3)
ct = ConductivityTensorFunction(
pc.conductivity)
Aie = matfac.computeStiffnessMatrix(ct)
# Construct compound matrices
dt = 0.1
A = M + dt*Ai
B = dt*Ai
Bt = dt*Ai
C = dt*Aie
# Create the Block system
AA = BlockMatrix((A,B),(Bt,C))
prec = DiagBlockMatrix((MLPrec(A),
MLPrec(C)))
v = numpy.zeros(A.n, dtype='d') - 45.0
u = numpy.zeros(A.n, dtype='d')
x = BlockVector(v,u)
# Create one ODE systems for each vertex
odesys = Courtemanche_ODESystem()
ode_solver = RKF32(odesys)
ionic = IonicODEs(A.n, ode_solver,
odesys)
# Solve
z = numpy.zeros((A.n,), dtype='d')
for i in xrange(0, 10):
t = i*dt
ionic.forward(x[0], t, dt)
ConjGrad.precondconjgrad(prec, AA,
x, BlockVector(M*x[0], z))
