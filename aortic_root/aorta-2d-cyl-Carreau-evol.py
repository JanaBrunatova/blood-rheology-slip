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

info(f"Using hematocrit {hematocrit}% with Carreau parameters: eta0={carreau_parameters[hematocrit]['eta0']}, etainf={carreau_parameters[hematocrit]['etainf']}, lambda={carreau_parameters[hematocrit]['lam']}, n_pow={carreau_parameters[hematocrit]['n_pow']}")

vtkoutput = True

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
pfile = XDMFFile(comm, f"results_Carreau_{hematocrit}/p.xdmf")
vfile = XDMFFile(comm, f"results_Carreau_{hematocrit}/v.xdmf")

pfile.parameters["flush_output"] = True
vfile.parameters["flush_output"] = True

# Problem parameters
## Timestep
dt = Constant(0.01)
t_end = 3.0


## Proportions
Len = 0.022
#Len_sin=0.012
Win = 0.012
Wout = 0.013


## Benchmark parameters
rho = 1050.0 #kg/m3
gamma = 3.08

## Carreau rheological parameters (used in both inlet profile and variational form)
eta0   = carreau_parameters[hematocrit]["eta0"]   # zero-shear viscosity [Pa s]
etainf = carreau_parameters[hematocrit]["etainf"] # infinite-shear viscosity [Pa s]
lam    = carreau_parameters[hematocrit]["lam"]    # time constant [s]
n_pow  = carreau_parameters[hematocrit]["n_pow"]  # shear-thinning exponent [-]
mu     = etainf     # reference viscosity for Nitsche penalty

info("Using Carreau parameters for hematocrit {}%: eta0={}, etainf={}, lambda={}, n={}".format(
    hematocrit, eta0, etainf, lam, n_pow))

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
# This is the parabolic profile for Newtonian fluid, but we will overwrite it with Carreau-Poiseuille profile below
#profile = "(v*(4.0*Gamma*Nu*(1.0-t)*r + 2.0*t*(pow(r,2)-pow(x[0],2))))/( 4.0*Gamma*Nu*(1-t)*r + t*r*r)"
#v_in = Expression((0, profile), r=Win, Gamma=gamma, Nu=mu, t=float(Theta), v=V, degree=2)

# --- Carreau-Poiseuille velocity profile at inlet ---
def compute_carreau_poiseuille(R, V_mean, eta0, etainf, lam, n_pow, kappa=None, N=2000):
    """
    Solve fully-developed pipe flow for Carreau fluid with Navier slip.
    For each r, solve  mu(0.5*gdot^2)*gdot = G*r/2  for gdot(r),
    then  v_z(r) = v_slip + int_r^R gdot(s) ds,
    where  v_slip = G*R/(2*kappa)  (zero for no-slip, i.e. kappa=None).
    Pressure gradient G is found so that mean velocity = V_mean.
    """
    def eta_carreau(gdot2_half):
        return etainf + (eta0 - etainf) * (1.0 + lam**2 * gdot2_half) ** ((n_pow - 1.0) / 2.0)

    r_vals = np.linspace(0, R, N + 1)

    def velocity_for_G(G):
        gdot_vals = np.zeros(N + 1)
        for i, ri in enumerate(r_vals):
            tau = G * ri / 2.0
            if tau < 1e-30:
                gdot_vals[i] = 0.0
            else:
                # solve eta(0.5*g^2)*g = tau  for g >= 0
                f = lambda g, _tau=tau: eta_carreau(0.5 * g * g) * g - _tau
                g_max = tau / etainf * 2.0
                gdot_vals[i] = brentq(f, 0.0, g_max)
        # v_z^{no-slip}(r) = integral from r to R of gdot(s) ds
        u_vals = np.zeros(N + 1)
        for i in range(N - 1, -1, -1):
            dr = r_vals[i + 1] - r_vals[i]
            u_vals[i] = u_vals[i + 1] + 0.5 * (gdot_vals[i] + gdot_vals[i + 1]) * dr
        # add Navier-slip shift: v_z(R) = G*R/(2*kappa)
        if kappa is not None and kappa > 0:
            u_vals += G * R / (2.0 * kappa)
        # mean velocity:  V = (2/R^2) int_0^R r v_z(r) dr
        U_mean = 2.0 / R**2 * np.trapz(r_vals * u_vals, r_vals)
        return u_vals, U_mean

    # bracket G so that V_mean is achieved
    G_newt = 8.0 * etainf * V_mean / R**2
    G_opt = brentq(lambda G: velocity_for_G(G)[1] - V_mean, G_newt * 0.001, G_newt * 1000.0)
    u_vals, U_actual = velocity_for_G(G_opt)
    slip_str = "kappa={:.4e}".format(kappa) if kappa is not None else "no-slip"
    info("Carreau-Poiseuille inlet ({}): G={:.6e}, V_mean={:.6e}, v_max={:.6e}".format(
        slip_str, G_opt, U_actual, u_vals[0]))
    return r_vals, u_vals, G_opt

# Navier-slip coefficient kappa taken directly from input
_kappa_val = float(Kappa)
if _kappa_val == 0.0:
    _kappa = 0.0   # perfect slip (kappa = 0) => plug flow
else:
    _kappa = _kappa_val

if _kappa == 0.0:
    # perfect slip: plug flow v_z = V everywhere
    _r_prof = np.linspace(0, Win, 2001)
    _u_prof = np.full_like(_r_prof, V)
    _G_poiseuille = 0.0
    info("Carreau-Poiseuille inlet (perfect slip): plug flow v_z = {:.6e}".format(V))
else:
    _r_prof, _u_prof, _G_poiseuille = compute_carreau_poiseuille(Win, V, eta0, etainf, lam, n_pow, kappa=_kappa)

# --- Reference quantities for Carreau-Poiseuille flow ---
_tau_w = _G_poiseuille * Win / 2.0
info("Reference: G = {:.6e}".format(_G_poiseuille))
info("Reference: tau_w = G*R/2 = {:.6e}".format(_tau_w))
if _kappa is not None and _kappa > 0:
    _v_slip = _G_poiseuille * Win / (2.0 * _kappa)
    info("Reference: v_slip = {:.6e}".format(_v_slip))
_u_interp = interp1d(_r_prof, _u_prof, kind='cubic', fill_value=0.0, bounds_error=False)

class CarreauPoiseuilleInflow(UserExpression):
    def update(self, V):
        if _kappa == 0:
           _G_poiseuille = 0.0
           self._u_interp = lambda r: V
        else:
           _r_prof, _u_prof, _G_poiseuille = compute_carreau_poiseuille(Win, V, eta0, etainf, lam, n_pow, kappa=_kappa)
           self._u_interp = interp1d(_r_prof, _u_prof, kind='cubic', fill_value=0.0, bounds_error=False)
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = float(self._u_interp(abs(x[0])))
    def value_shape(self):
        return (2,)

v_in = CarreauPoiseuilleInflow(degree=2)

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

def gammadot2(v):
    return (inner(D(v),D(v))+v[0]*v[0]/(rd*rd))

def nu(gammadot2):
    # Carreau model for blood (parameters defined above)
    return etainf + (eta0-etainf)*((1.0+lam*lam*gammadot2)**((n_pow-1.0)/2.0))

def CauchyT(p,v):
    return -p*I + 2.0*nu(gammadot2(v))*D(v)

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
Eq2 = inner(rho*rd*((v-v0)/dt + convterm(v)), v_)*dx + rd*inner(CauchyT(p,v),grad(v_))*dx - p*v_[0]*dx + (2.0*nu(gammadot2(v))*v[0]/rd)*v_[0]*dx
Eq3 = (-0.5*rd*rho*negpart(inner(v,n['out']))*inner(v,v_))*ds(marks['out']) #outflow
Eq4 = (Kappa*inner(vt(v,n['wall']),vt(v_,n['wall'])))*rd*ds(marks['wall']) #wall
Eq56 = - inner(flux, derivative(inner(v,n['wall']), w, w_))*rd*ds(marks['wall']) + inner(derivative(flux, w, w_), inner(v,n['wall']))*rd*ds(marks['wall'])
#Eq5 = (-inner(dot(CauchyT(p,v),n['wall']),n['wall'])*inner(v_,n['wall']))*rd*ds(marks['wall']) #wall
#Eq6 = (inner(dot(CauchyT(p_,v_),n['wall']),n['wall'])*inner(v,n['wall']))*rd*ds(marks['wall']) #wall
Eq7 = beta*mu/edgelen*inner(v,n['wall'])*inner(v_,n['wall'])*rd*ds(marks['wall'])

Eq = Eq1 + Eq2 + Eq3 + Eq4 + Eq56 + Eq7

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
solver.parameters['newton_solver']['maximum_iterations'] = 16
solver.parameters['newton_solver']["krylov_solver"]['error_on_nonconvergence'] = False

# Time stepping
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

DG0 = FunctionSpace(mesh, FiniteElement("DG", mesh.ufl_cell(), 0))
VCG1 = FunctionSpace(mesh, VectorElement("CG", mesh.ufl_cell(), 1))
CG1 = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1))

def extract_quantites(domain_parts, bndry, Len, mesh) :
    r=dict()

    curlv = Function(CG1)
    curlv.assign(project(v[0].dx(1)-v[1].dx(0), CG1,solver_type='mumps'))

    volume = 2.0*pi*assemble(interpolate(Constant(1.0),DG0)*rd*dx)
    area_wall = 2.0*pi*assemble(interpolate(Constant(1.0),DG0)*rd*ds(marks['wall']))
    #info("area_wall = {}".format(float(area_wall)))
    r['bulk_diss'] = 2.0*pi*assemble(2.0*nu(gammadot2(v))*(rd*inner(D(v),D(v))+v[0]*v[0]/rd)*dx)
    r['bndry_diss'] = 2.0*pi*Kappa*assemble(rd*inner(vt(v,n['wall']),vt(v,n['wall']))*ds(marks['wall']))
    r['total_diss'] = r['bulk_diss'] + r['bndry_diss']
    r['WSScomp'] = 2.0*pi*Kappa*assemble(sqrt(inner(vt(v,n['wall']),vt(v,n['wall'])))*rd*ds(marks['wall']))/area_wall
    #r['Pdrop'] =  parallel_eval(p,[0,-Len]) - parallel_eval(p,[0,Len])  # p_in-p_out
    r['Pdrop'] =  p(0, -Len) - p(0, Len)  # p_in-p_out
    r['Vort'] = 2.0*pi*assemble(abs(curlv)*rd*dx)/volume
    r['normalvel'] = (2.0*pi*assemble(inner(v, n['wall'])*inner(v, n['wall'])*rd*ds(marks['wall'])))**(0.5)
    return(r)


while t < t_end + 1e-6:
    info("t = {}".format(t))

    V_mean = inflow(t)
    v_in.update(V_mean)
    info("inflow velocity = {}".format(float(V_mean)))   

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

    rfull = extract_quantites(mesh, boundary_parts, Len, mesh)

    if rank == 0:
        output_file = open(filename, "a")
        output_file.write(str(float(t)) + " " + str(float(Kappa)) + " " + str(float(rfull['bulk_diss'])) + " " + str(float(rfull['bndry_diss'])) + " " + str(float(rfull['total_diss'])) + " " + str(float(rfull['WSScomp'])) + " " + str(float(rfull['Pdrop'])) + " " + str(float(rfull['Vort'])) + " " + str(float(rfull['normalvel'])) +  "\n")
        output_file.flush()
        output_file.close()


    w0.assign(w)
    t = round(float(t + dt), 3)
    info("dt = {}".format(float(dt)))

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
