from petsc4py import PETSc
print=PETSc.Sys.Print

import math 
import numpy as np

from dolfin import *

opts = PETSc.Options()

def my_solve(A,x,b):
    ksp = PETSc.KSP().create()
    ksp.setType('preonly')
    pc =PETSc.PC().create()
    pc.setType('lu')
    pc.setFactorSolverType("superlu_dist")
    ksp.setPC(pc)
    
    ksp.setOperators(as_backend_type(A).mat())
    ksp.setTolerances(rtol=1e-20, atol=1e-10, max_it=1000)

    print('Solving with:', ksp.getType(), pc.getType(), flush=True)
    # Solve
    bb=as_backend_type(b).vec()
    xx=as_backend_type(x).vec()
    ksp.solve(bb, xx)
    print("Converged reason:",ksp.getConvergedReason(), " in ",ksp.getIterationNumber(), 'iterations', flush=True)

def normalize_vector(n):
    V=n.function_space()
    nx_dofs=V.sub(0).dofmap().dofs()
    ny_dofs=V.sub(1).dofmap().dofs()

    nv=as_backend_type(n.vector()).vec()

    nx=nv[nx_dofs]
    ny=nv[ny_dofs]
    dn=np.sqrt(nx*nx+ny*ny)

    nx = np.divide(nx,dn, where=(dn>0.0))
    ny = np.divide(ny,dn, where=(dn>0.0))
    n.vector().update_ghost_values()

    
#compute normal vector on boundary marked by id, by projection of FacetNormal to FE space
def make_normal_projection(mesh, bndry, id=None, type='CG1'):
    print(f"Computing {type} normal.... {id}")
    if type=='DG0' :
        degree=1
        ve=VectorElement("CR", mesh.ufl_cell(), 1)
        e=FiniteElement("CR", mesh.ufl_cell(), 1)
    elif type=='CG1' :
        degree=1
        ve=VectorElement("CG", mesh.ufl_cell(), 1)
        e=FiniteElement("CG", mesh.ufl_cell(), 1)
    else :
        raise ValueError("Invalid normal type.")
    
    V=FunctionSpace(mesh, e)
    VV=FunctionSpace(mesh, ve)

    n=FacetNormal(mesh)
    ds = Measure("ds", subdomain_data=bndry, domain=mesh)
    
    u, v = TrialFunction(V), TestFunction(V)
    nn = Function(V)
    vn = Function(VV)
    
    a = u*v*ds(id) + Constant(0.0)*avg(u)*avg(v)*dS 
    A = assemble(a, keep_diagonal=True)
    A.ident_zeros()

    ksp = PETSc.KSP().create()
    ksp.setType('preonly')
    pc =PETSc.PC().create()
    pc.setType('lu')
    pc.setFactorSolverType("superlu_dist")
    ksp.setPC(pc)

    ksp.setOperators(as_backend_type(A).mat())
    ksp.setTolerances(rtol=1e-20, atol=1e-10, max_it=1000)

    print('Solving with:', ksp.getType(), pc.getType(), flush=True)
    # Solve
    # make it faster by computing each component separately (reuse LU factors)
    for i in [0,1] :
        L = inner(n[i], v)*ds(id)
        b = assemble(L)
        nn.vector().zero()

        bb=as_backend_type(b).vec()
        xx=as_backend_type(nn.vector()).vec()
        ksp.solve(bb, xx)
        nn.vector().update_ghost_values()
        print("Converged reason:",ksp.getConvergedReason(), " in ",ksp.getIterationNumber(), 'iterations', flush=True)

        dofs=VV.sub(i).dofmap().dofs()
        as_backend_type(vn.vector()).vec()[dofs]=as_backend_type(nn.vector()).vec()
        vn.vector().update_ghost_values()
        vn.vector().apply("insert")
            
    normalize_vector(vn)
    print(f"done.")
    return(vn)
