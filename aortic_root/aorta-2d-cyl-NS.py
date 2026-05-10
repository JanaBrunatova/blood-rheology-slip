from fenics import *
from mshr import *
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import sys
if len(sys.argv)<=4:
   info("Input parameters are missing. The first parameter is kappa (Navier-slip coefficient), the second parameter is filename output, the third parameter is penalty beta, the fourth parameter is sinus radius, the fifth parameter is Hct level.")
   sys.exit()

Kappa = Constant(float(sys.argv[1]))
filename = sys.argv[2]
beta = Constant(float(sys.argv[3]))
radius = int(sys.argv[4])

# Hematocrit setting.
# By default use 45%, optionally override by the 5th command-line argument.
hematocrit = 45
if len(sys.argv) >= 6:
    hematocrit = int(sys.argv[5])

carreau_parameters = {
    25: {"eta0": 0.0178, "lam": 12.448, "n_pow": 0.330, "etainf": 0.00257},
    45: {"eta0": 0.1610, "lam": 39.418, "n_pow": 0.479, "etainf": 0.00345},
    65: {"eta0": 0.8592, "lam": 103.088, "n_pow": 0.389, "etainf": 0.00802},
}

if hematocrit not in carreau_parameters:
    info("Unsupported hematocrit '{}'. Use one of 25, 45, 65.".format(hematocrit))
    sys.exit()

info(f"Using hematocrit {hematocrit} with viscosity parameter mu={carreau_parameters[hematocrit]['etainf']}")

vtkoutput = False

PETScOptions.set('mat_mumps_icntl_14', 1000) # work array, multiple of estimate to allocate
PETScOptions.set('mat_mumps_icntl_24', 1)  # detect null pivots
PETScOptions.set('mat_mumps_cntl_1', 1.0)  # pivoting threshold, this solves to machine precision

comm = MPI.comm_world
rank = MPI.rank(comm)
set_log_level(LogLevel.INFO if rank==0 else LogLevel.INFO)
parameters["std_out_all_processes"] = False
parameters["form_compiler"]["quadrature_degree"] = 8
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["ghost_mode"] = "shared_facet"

# Save solution in VTK format
pfile = XDMFFile(comm, "results/p.xdmf")
vfile = XDMFFile(comm, "results/v.xdmf")

pfile.parameters["flush_output"] = True
vfile.parameters["flush_output"] = True

# Problem parameters
## Timestep
dt = Constant(0.01)
t_end = 1.0e6


## Proportions
Len = 0.022
Win = 0.012


## Benchmark parameters
rho = 1050.0 #kg/m3

## Carreau rheological parameters (used in both inlet profile and variational form)
eta0   = carreau_parameters[hematocrit]["eta0"]   # zero-shear viscosity [Pa s]
etainf = carreau_parameters[hematocrit]["etainf"] # infinite-shear viscosity [Pa s]
lam    = carreau_parameters[hematocrit]["lam"]    # time constant [s]
n_pow  = carreau_parameters[hematocrit]["n_pow"]  # shear-thinning exponent [-]
mu     = etainf     # reference viscosity for Nitsche penalty

mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), f"mesh_R{radius}.h5", "r")
#info(hdf.parameters, True)
hdf.read(mesh, "/mesh", False)
mesh.init()

# Prepare boundaries
boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim()-1,0)

# Define finite elements
Ev = VectorElement("CG", mesh.ufl_cell(), 2)    # Velocity (vr, vz)
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)    # Pressure p

W = FunctionSpace(mesh, MixedElement([Ev, Ep])) # order of elements matters

marks={'in': 1, 'out': 2, 'wall': 3, 'symm': 4}

# Define boundaries
for f in facets(mesh):
    mp = f.midpoint()
    if f.exterior() : boundary_parts[f] = marks['wall']  # wall
    # overwrite straight boundaries
    if near(mp[1], -Len): # inflow
        boundary_parts[f] = marks['in']
    elif near(mp[1], Len): # outflow
        boundary_parts[f] = marks['out']
    #elif mp[0] >= (Wid-0.001) and f.exterior(): # wall
    #    boundary_parts[f] = 3
    elif near(mp[0], 0): # wall of symmetry
        boundary_parts[f] = marks['symm']

#### generate projection normal
from generate_normal_2D import make_normal_projection
n=dict()
for i in ['wall','in','out','symm']:
   n[i]=make_normal_projection(mesh, boundary_parts, id=marks[i]) 

        
## Facet normal and boundary measure
ds = Measure("ds", subdomain_data=boundary_parts)
h = CellDiameter(mesh)

# Define boundary conditions
bc_symmetry_wall = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundary_parts, 4)

V = 0.65
# Newtonian Poiseuille profile with Navier-slip kappa:
#   v(r) = V * (K*(R^2 - r^2) + 2*Nu*R) / (K*R^2/2 + 2*Nu*R)
# No-slip limit (K->inf): parabola; perfect-slip (K=0): plug flow.
profile = "v * (K*(r*r - x[0]*x[0]) + 2.0*Nu*r) / (K*r*r/2.0 + 2.0*Nu*r)"
v_in = Expression((0, profile), r=Win, Nu=mu, K=float(Kappa), v=V, degree=2)

bc_inflow = DirichletBC(W.sub(0), v_in, boundary_parts, 1)
bc_outflow = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundary_parts, 2)

bcs = [bc_symmetry_wall, bc_inflow, bc_outflow]

# Test functions
w_ = TestFunction(W)
v_, p_ = split(w_)
w = Function(W)
v,  p = split(w)

#Previous time step
w0 = Function(W)
(v0, p0) = split(w0)

#Initial data
w0ic = Expression(("0.0","0.0","0.0"), degree = 1)
w0.assign(interpolate(w0ic, W))
w.assign(interpolate(w0ic, W))


# Define variational problem 
I = Identity(mesh.geometry().dim())
matderv = (v-v0)/dt + grad(v)*v

#rd = Expression("x[0]+1.e-10", degree=2)
rd = Expression("x[0]", degree=2)
#rd = Expression("sqrt(x[0]*x[0]+1.e-12)", degree=2)

def convterm(v):
    return grad(v)*v

def D(v):
    return 0.5*(grad(v)+grad(v).T)

def CauchyT(p,v):
    return -p*I + 2.0*mu*D(v)

def rdiv_axi(v):
    return rd*div(v) + v[0]

def negpart(s):
    return conditional(gt(s, 0.0), 0.0, 1.0)*s

def vn(v,n):
    return inner(v,n)*n

def vt(v,n):
    return v - vn(v,n)

edgelen = MPI.min(comm, mesh.hmin())

flux = inner(CauchyT(p,v)*n['wall'], n['wall'])

Eq1 = rdiv_axi(v)*p_*dx
Eq2 = inner(rho*rd*((v-v0)/dt + convterm(v)), v_)*dx + rd*inner(CauchyT(p,v),grad(v_))*dx - p*v_[0]*dx + (2.0*mu*v[0]/rd)*v_[0]*dx
Eq3 = (-0.5*rd*rho*negpart(inner(v,n['out']))*inner(v,v_))*ds(marks['out']) #outflow
Eq4 = (Kappa*inner(vt(v,n['wall']),vt(v_,n['wall'])))*rd*ds(marks['wall']) #wall
Eq5 = (-inner(dot(CauchyT(p,v),n['wall']),n['wall'])*inner(v_,n['wall']))*rd*ds(marks['wall']) #wall
Eq6 = (inner(dot(CauchyT(p_,v_),n['wall']),n['wall'])*inner(v,n['wall']))*rd*ds(marks['wall']) #wall
Eq7 = beta*mu/edgelen*inner(v,n['wall'])*inner(v_,n['wall'])*rd*ds(marks['wall'])

Eq = Eq1 + Eq2 + Eq3 + Eq4 + Eq5 + Eq6 + Eq7

## prepare solver 
info("Kappa: {}".format(float(Kappa)))
info("Solving problem of size: {0:d}".format(W.dim()))
J=derivative(Eq,w)
problem=NonlinearVariationalProblem(Eq,w,bcs,J)
solver=NonlinearVariationalSolver(problem)
## set solvers parameters
solver.parameters['newton_solver']['error_on_nonconvergence'] = False #continue if diverged 
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-13
solver.parameters['newton_solver']['relative_tolerance'] = 1e-13
solver.parameters['newton_solver']['maximum_iterations'] = 10
solver.parameters['newton_solver']["krylov_solver"]['error_on_nonconvergence'] = False

# Time stepping
optimal_it = 9.001
dt_max = 500000.0
t = float(dt)

def inflow(t):
    T = 1.0
    x0 = 0.3
    V = 0.7
    tt = t % T
    if(tt<x0):
      return -4.0*V/(x0*x0)*tt*(tt-x0)
    else:
      return 0.0


# Time stepping
optimal_it = 9.001
dt_max = 500000.0
t = float(dt)


while t < t_end:
    info("t = {}".format(t))

    # Solving problem
    its, ok = solver.solve()

    # "uncouple" solution to separate pressure and velocity
    v, p = w.split(True) # "True" creates deep copy
    v.rename("Velocity", "v")
    p.rename("Pressure","p")

    #info("Area: {}".format(area_out))
    # File output
    if vtkoutput:
       pfile.write(p,t)
       vfile.write(v,t)

    #dt.assign(time_step(v)) # Depends on time, has to be assign
    failed = False
    if (ok == 0):
       info("Step back!")
       t -= float(dt)/2.0
       dt.assign(float(dt)/2.0)
       info("dt = {}".format(float(dt)))
       w.assign(w0)
       failed = True
       continue

    if its <= optimal_it:
       dt.assign(min(dt_max, float(dt)*min(4.0,optimal_it/(its+0.001))))
    else:
       dt.assign(float(dt)*optimal_it/its)
    w0.assign(w)

    # Move to next time step
    t += float(dt)

    info("dt = {}".format(float(dt)))


DG0 = FunctionSpace(mesh, FiniteElement("DG", mesh.ufl_cell(), 0))
VCG1 = FunctionSpace(mesh, VectorElement("CG", mesh.ufl_cell(), 1))
CG1 = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1))

curlv = Function(CG1)
curlv.assign(project(v[0].dx(1)-v[1].dx(0), CG1,solver_type='mumps'))

def parallel_eval(f, x):
    mesh = f.function_space().mesh()
    bb = mesh.bounding_box_tree()

    p=Point(x)
    value=np.array([0.0])
    ic=0
    cf=bb.compute_first_entity_collision(p)
    inside= cf < mesh.num_cells()
    if inside :
        f.eval_cell(value,x,Cell(mesh,cf))
        ic=1
       
    comm=MPI.comm_world
    v= MPI.sum(comm, value) / MPI.sum(comm, ic)
    return(v)

def extract_quantites(domain_parts, bndry, Len, mesh) :

    r=dict()

    volume = 2.0*pi*assemble(interpolate(Constant(1.0),DG0)*rd*dx)
    area_wall = 2.0*pi*assemble(interpolate(Constant(1.0),DG0)*rd*ds(marks['wall']))
    #info("area_wall = {}".format(float(area_wall)))
    r['bulk_diss'] = 2.0*pi*assemble(2.0*mu*(rd*inner(D(v),D(v))+v[0]*v[0]/rd)*dx)
    r['bndry_diss'] = 2.0*pi*float(Kappa)*assemble(rd*inner(vt(v,n['wall']),vt(v,n['wall']))*ds(marks['wall']))
    r['total_diss'] = r['bulk_diss'] + r['bndry_diss']
    r['WSScomp'] = 2.0*pi*float(Kappa)*assemble(sqrt(inner(vt(v,n['wall']),vt(v,n['wall'])))*rd*ds(marks['wall']))/area_wall
    #r['Pdrop'] =  parallel_eval(p,[0,-Len]) - parallel_eval(p,[0,Len])  # p_in-p_out
    r['Pdrop'] =  p(0, -Len) - p(0, Len)  # p_in-p_out
    r['Vort'] = 2.0*pi*assemble(abs(curlv)*rd*dx)/volume
    r['normalvel'] = (2.0*pi*assemble(inner(v, n['wall'])*inner(v, n['wall'])*rd*ds(marks['wall'])))**(0.5)
    return(r)

rfull = extract_quantites(mesh, boundary_parts, Len, mesh)

if rank == 0:
    output_file = open(filename, "a")
    output_file.write(str(float(t)) + " " + str(float(Kappa)) + " " + str(float(rfull['bulk_diss'])) + " " + str(float(rfull['bndry_diss'])) + " " + str(float(rfull['total_diss'])) + " " + str(float(rfull['WSScomp'])) + " " + str(float(rfull['Pdrop'])) + " " + str(float(rfull['Vort'])) + " " + str(float(rfull['normalvel'])) +  "\n")
    output_file.flush()
    output_file.close()


pfile.close()
vfile.close()

del p
del v
del w
del w0

import gc
gc.collect()

print("All saved, exiting.")
import os
os._exit(0)
