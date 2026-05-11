import argparse
import os
import re

import dolfin as df
import numpy as np
from ufl import block_split

from ns_aneurysm import finite_elements, inflow, mesh_info, model

comm = df.MPI.comm_world
rank = df.MPI.rank(comm)

desc = "Compute wall shear stress given a HDF5 file of the velocity and pressure (v, p)"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=desc
)
parser.add_argument(
    "--mesh",
    default="../meshes/cylinder_uniform_marked.h5",
    type=str,
    dest="mesh_file",
    help="Path to a marked mesh file",
)
parser.add_argument(
    "--element",
    default="th",
    type=str,
    choices=["p1p1", "mini", "th"],
    dest="elem",
    help="finite element; mini or th",
)
parser.add_argument(
    "--stab",
    default="none",
    type=str,
    choices=["ip", "supg", "none"],
    dest="stab",
    help="stabilization; ip or supg or none",
)
parser.add_argument(
    "-o",
    default="results_test/",
    type=str,
    dest="res_folder",
    help="results folder",
)
parser.add_argument(
    "--marker-wall",
    default=1,
    type=int,
    dest="mark_wall",
    help="Integer describing which marker is at the walls, i.e the boundary to compute WSS on",
)
parser.add_argument(
    "--refsystems-file",
    default="",
    type=str,
    dest="refsystems_file",
    help="a path to the file with reference systems corresponding to the marked mesh (from vmtk .dat)",
)
parser.add_argument(
    "--mu",
    default=4e-3,
    type=float,
    dest="mu",
    help="Dynamic viscosity in SI units [Pa.s]. Default value is 4 mPa.s",
)
parser.add_argument(
    "--rho",
    default=1000,
    type=float,
    dest="rho",
    help="Density in SI units [kg/m^3]. Default value is 1000 kg/m^3",
)
parser.add_argument(
    "--beta",
    default=10,
    type=float,
    dest="beta",
    help="Parameter for the Nitsche BC. Default value is 10.",
)
parser.add_argument(
    "--w-file",
    default="",
    type=str,
    dest="w_file",
    help="Path to our computed velocity and pressure",
)
parser.add_argument(
    "--compute-differences",
    action="store_true",
    default=False,
    dest="compute_differences",
    help="If true, differences between maximum values of WSS for different FE spaces will be computed.",
)
parser.add_argument(
    "--stationary",
    action="store_true",
    default=False,
    dest="stationary",
    help="Were these results computed for a stationary or a pulsatile flow?",
)
parser.add_argument(
    "--nitsche-noslip",
    action="store_true",
    default=False,
    dest="nitsche_noslip",
    help="if True, Nitsche terms will be added to the form for computing WSS",
)
parser.add_argument(
    "--model",
    default="Newtonian",
    type=str,
    dest="model_name_str",
    help="Name of the viscosity model from model.py",
)
parser.add_argument(
    "--theta",
    default=-1.0,
    type=float,
    dest="theta",
    help="Slip parameter. -1 for Dirichlet no-slip, or [0, 1] for Nitsche BC slip.",
)
parser.add_argument(
    "--t-begin",
    "--t_begin",
    default=0.0,
    type=float,
    dest="t_begin",
    help="Start time for WSS postprocessing. Timesteps with timestamp < t_begin are skipped.",
)

args = parser.parse_args()
mark_wall = args.mark_wall
mu = args.mu
rho = args.rho
elem = args.elem
stab = args.stab
mesh_file = args.mesh_file
refsystems_file = args.refsystems_file
w_file = args.w_file
stationary = args.stationary
model_name_str = args.model_name_str
theta = args.theta
t_begin = args.t_begin

comm = df.MPI.comm_world

# read mesh
mesh = df.Mesh()
with df.HDF5File(comm, mesh_file, "r") as hdf:
    hdf.read(mesh, "/mesh", False)
    dim = mesh.geometry().dim()
    boundary_parts = df.MeshFunction("size_t", mesh, dim - 1, 0)
    hdf.read(boundary_parts, "/boundaries")
mesh.init()


def T(p, v):  # Cauchy stress tensor
    return -p * df.Identity(mesh.geometry().dim()) + 2.0 * visc_model.viscosity * df.sym(
        df.grad(v)
    )


def tangential_proj(v, n):
    """
    Compute the tangential projection of a vector v given the normal vector n.
    """
    return (df.Identity(v.ufl_shape[0]) - df.outer(n, n)) * v


class TractionComputation:
    def __init__(
        self,
        traction_space,
        mark_wall: int = 1,
        direct_solver: bool = False,
    ):
        self.mark_wall = mark_wall
        self.space = traction_space

        # Set quantities to compute the WSS
        x = df.TrialFunction(self.space)
        self.x_ = df.TestFunction(self.space)
        lhs = df.inner(x, self.x_) * ds(mark_wall)

        A = df.assemble(lhs, keep_diagonal=True)
        A.ident_zeros()

        if direct_solver:
            self.solver_wss = df.LUSolver("mumps")
        else:
            self.solver_wss = df.KrylovSolver("bicgstab", "jacobi")

        self.solver_wss.set_operator(A)

    def project_function(self, b: df.Function) -> df.Function:
        """
        Compute traction force as a projection of a function b.
        """
        traction = df.Function(self.space)
        Ln = df.inner(b, self.x_) * ds(self.mark_wall)
        rhs = df.assemble(Ln)
        self.solver_wss.solve(traction.vector(), rhs)
        return traction

    def project_wss(self, b, n):
        """
        Compute wall shear stress as a projection of a function b.
        """
        wss = df.Function(self.space)
        b_t = tangential_proj(b, n)
        Ln = df.inner(b_t, self.x_) * ds(self.mark_wall)
        rhs = df.assemble(Ln)
        self.solver_wss.solve(wss.vector(), rhs)
        return wss

    def project_rescaled_velocity(self, v: df.Function, theta: float) -> df.Function:
        """
        Compute WSS purely as a rescaled velocity field when theta != 1 and theta != -1.
        Assumes impermeable walls (v.n = 0), making v entirely tangential.
        """
        scaling_factor = theta / (theta - 1.0)
        rescaled_v = df.Constant(scaling_factor) * v
        return self.project_function(rescaled_v)


def compute_norm(data, result=None):
    """
    input: quantity living in some VectorElement space (for example velocity or wall shear stress vector)
    output: norm of the quantity living in appropriate FunctionSpace
    """
    V_local = data.function_space()
    nx_dofs = V_local.sub(0).dofmap().dofs()
    ny_dofs = V_local.sub(1).dofmap().dofs()
    nz_dofs = V_local.sub(2).dofmap().dofs()

    nx = data.vector().vec()[nx_dofs]
    ny = data.vector().vec()[ny_dofs]
    nz = data.vector().vec()[nz_dofs]
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)

    if result is None:
        S = V_local.sub(0).collapse()
        result = df.Function(S)

    result.vector().set_local(norm)
    result.vector().apply("insert")
    return result


def pvn(v, n):  # normal component of a vector v
    return df.inner(v, n) * n


def vn(v, n):  # projection of a vector v to the direction of normal vector
    return df.inner(v, n)


def pvt(v, n):  # tangential component of a vector v
    return v - df.inner(v, n) * n


# get finite element spaces
ns_element = getattr(finite_elements, elem)
FE = ns_element(mesh, boundary_parts)
W = FE.W
V = W.sub(0).collapse()
P = W.sub(1).collapse()
dx = FE.dx(metadata={"quadrature_degree": 4})
ds = FE.ds(metadata={"quadrature_degree": 4})
dS = FE.dS(metadata={"quadrature_degree": 4})

# ==================================
# create xdmf files
xdmf_keys = [
    "v",
    "v_cp",
    "p_cp",
    "traction_weak",
    "wss_weak_CG_1_cp",
    "wss_weak_CG_2_cp",
    "wss_standard_DG_0",
    "wss_standard_DG_1",
    "wss_standard_CG_1",
    "wss_standard_DG_0_cp",
    "wss_standard_DG_1_cp",
    "wss_standard_CG_1_cp",
    "wss_rescaled_v_tan",
    "wss_rescaled_v_tan_cp",
]

file_xdmf = dict()
for i in xdmf_keys:
    file_xdmf[i] = df.XDMFFile(comm, args.res_folder + i + ".xdmf")
    file_xdmf[i].parameters["flush_output"] = True
    file_xdmf[i].parameters["rewrite_function_mesh"] = False

# Define normal field
n = df.FacetNormal(mesh)

w1 = df.Function(W)
(v, p, v_, p_) = FE.split(w1)

# Set viscosity model from model.py
model_name = getattr(model, model_name_str)
scale_CGS_to_SI = 0.1  # Poise to SI

if model_name_str == "Newtonian":
    visc_model = model_name(mesh, v, scale=1.0, mu=mu)
else:
    visc_model = model_name(mesh, v, scale=scale_CGS_to_SI)

wdot = df.Function(W)

if rank == 0:
    print(f"reading file {w_file}")

fh5 = df.HDF5File(comm, w_file, "r")
ntimesteps = fh5.attributes("/w")["count"]

h = df.Constant(2.0) * df.Circumradius(mesh)
h2 = df.Constant(2.0) * df.Circumradius(mesh)


def NitscheBC(eq, n, ds):
    """
    Nitsche's method implemented through the derivative of a functional.
    """
    w_ = df.TestFunction(w1.function_space())
    penalty = (args.beta * mu / h) * df.inner(eq, df.derivative(eq, w1, w_)) * ds
    bcpart = (
        df.inner(T(p, v) * n, df.derivative(eq, w1, w_)) * ds
        - df.inner(df.derivative(T(p, v) * n, w1, w_), eq) * ds
    )
    return -bcpart + penalty


match = re.search(r"(case\d{2})", mesh_file)
case = match.group(1)

if refsystems_file == "":
    refsystems_file = f"meshes/{case}_refsystems_SI.dat"

refsys = mesh_info.read_file(refsystems_file)  # output from vmtk

# identify inflow id as the end with largest diameter
rr = 0.0
inflow_idx = 0
for i in range(len(refsys)):
    if refsys[i].r > rr:
        rr = refsys[i].r
        inflow_idx = i

# select the remaining ids as outflow
outflow_idx = []
for i in range(len(refsys)):
    if (i != inflow_idx) and (refsys[i].r > 0.0):
        outflow_idx.append(i)

# mark boundaries
mark_in = 2  # since 0 is mark for interior facets and 1 for walls
marks_out = list(range(3, len(refsys) + 2))

mesh_in_out = []
j = 0
for i in range(len(refsys)):
    nn = (refsys[i].nx, refsys[i].ny, refsys[i].nz)
    r = refsys[i].r
    s = (refsys[i].sx, refsys[i].sy, refsys[i].sz)

    if i == inflow_idx:
        mark = mark_in
    else:
        mark = marks_out[j]
        j = j + 1

    mesh_in_out.append(mesh_info.MeshInOut(n=nn, s=s, r=r, mark=mark))


# ==================================
# assemble the LHS

a = df.Function(W)
(_, _, v_test, p_test) = FE.split(a)

# ensure P1 recovery for TH element
V_recovery = df.VectorFunctionSpace(mesh, "P", 1)

g = df.TrialFunction(V_recovery)
g_test = df.TestFunction(V_recovery)

lhs = df.inner(g, g_test) * ds(mark_wall)

A = df.assemble(lhs, keep_diagonal=True)
A.ident_zeros()

solver_wss = df.KrylovSolver("bicgstab", "jacobi")
solver_wss.set_operator(A)

if elem == "th":
    # ensure P2 recovery for TH element
    V_P2 = W.sub(0).collapse()

    g_P2 = df.TrialFunction(V_P2)
    g_test_P2 = df.TestFunction(V_P2)

    lhs_P2 = df.inner(g_P2, g_test_P2) * ds(mark_wall)

    A_P2 = df.assemble(lhs_P2, keep_diagonal=True)
    A_P2.ident_zeros()

    solver_wss_P2 = df.KrylovSolver("bicgstab", "jacobi")
    solver_wss_P2.set_operator(A_P2)


def get_timestep_time(i: int) -> float:
    """
    Read the timestamp of timestep i from the HDF5 file.
    """
    attrs = fh5.attributes(f"/w/vector_{i}").to_dict()
    t = attrs.get("timestamp")

    if t is None:
        raise RuntimeError(f"No timestamp found for /w/vector_{i}")

    return float(t)


def evaluate_wss_timestep(i: int, append: bool = False):
    """
    Evaluate WSS for timestep i.

    If the timestamp is smaller than t_begin, the timestep is skipped and
    nothing is written to the XDMF files.

    Returns
    -------
    wrote : bool
        True if the timestep was written, False if it was skipped.
    t : float
        Physical timestamp of the timestep.
    """
    t = get_timestep_time(i)

    if t < t_begin:
        if rank == 0:
            print(f"Skipping timestep {i}, t = {t:.6g} < t_begin = {t_begin:.6g}")
        return False, t

    fh5.read(w1, f"/w/vector_{i}")
    fh5.read(wdot, f"/wdot/vector_{i}")

    v_export = w1.sub(0, deepcopy=True)
    p_export = w1.sub(1, deepcopy=True)
    vdot = wdot.sub(0, deepcopy=True)

    visc_model.project_viscosity()
    v_export.rename("v", "velocity")

    file_xdmf["v_cp"].write_checkpoint(v_export, "velocity", t, append=append)
    file_xdmf["p_cp"].write_checkpoint(p_export, "pressure", t, append=append)
    file_xdmf["v"].write(v_export, t)

    # If theta is not 1.0 or -1.0, bypass weak and standard calculations
    if args.theta != 1.0 and args.theta != -1.0:
        traction_computer_rescaled = TractionComputation(V, mark_wall=mark_wall)
        wss_rescaled = traction_computer_rescaled.project_rescaled_velocity(
            pvt(v_export, n), args.theta
        )

        file_xdmf["wss_rescaled_v_tan_cp"].write_checkpoint(
            wss_rescaled, "wss", t, append=append
        )
        wss_rescaled.rename("wss", "wss")
        file_xdmf["wss_rescaled_v_tan"].write(wss_rescaled, t)

    # Compute WSS weakly
    F_wss = (
        rho * df.inner(vdot, v_test) * dx
        + rho * df.inner(df.grad(v) * v, v_test) * dx
        + df.inner(T(p, v), df.grad(v_test)) * dx
        + df.div(v) * p_test * dx
        - df.inner(T(p, v) * n, v_test) * ds(mark_in)
    )

    # add directional do-nothing BC
    for j in marks_out:
        F_wss += -df.inner(T(p, v) * n, v_test) * ds(j)

    # add nitsche terms
    if args.nitsche_noslip:
        F_wss += df.inner(
            T(p_test, v_test) * n, v - df.Constant((0.0, 0.0, 0.0))
        ) * ds(mark_wall)

        F_wss += (
            (args.beta * mu / h)
            * df.inner(v - df.Constant((0.0, 0.0, 0.0)), v_test)
            * ds(mark_wall)
        )

        v_in = inflow.InflowAnalyticalStac(
            mesh_in_out[inflow_idx].s,
            mesh_in_out[inflow_idx].r,
            mesh_in_out[inflow_idx].n,
            v_mean=0.5,
        )

        F_wss -= NitscheBC(v - v_in, n, ds(mark_in))

    if stab == "ip":
        alpha_i = 1e-3 * rho
        alpha_v = 1e-3 * rho
        alpha_p = 1.0 / rho

        F_stab = (
            alpha_i
            * df.avg(h2) ** 2
            * pow(df.dot(v("+"), n("+")), 2)
            * df.inner(df.jump(df.grad(v)), df.jump(df.grad(v_test)))
            * dS
        )
        F_stab += (
            alpha_v
            * df.avg(h2) ** 2
            * df.inner(df.jump(df.grad(v)), df.jump(df.grad(v_test)))
            * dS
        )
        F_stab += (
            alpha_p
            * df.avg(h2) ** 2
            * df.inner(df.jump(df.grad(p)), df.jump(df.grad(p_test)))
            * dS
        )
        F_wss += F_stab

    WFv = block_split(F_wss, 0)
    rhs = df.assemble(df.action(WFv, g_test))

    traction_weak = df.Function(V_recovery, name="traction_force")
    solver_wss.solve(traction_weak.vector(), rhs)

    file_xdmf["traction_weak"].write_checkpoint(
        traction_weak, "traction_force", t, append=append
    )

    F_wss_tan = F_wss - (
        df.inner(df.inner(T(p, v) * n, n) * n, v_test) * ds(mark_wall)
    )

    WFv = block_split(F_wss_tan, 0)
    rhs = df.assemble(df.action(WFv, g_test))

    traction_weak_tan = df.Function(V_recovery, name="wss_weak")
    solver_wss.solve(traction_weak_tan.vector(), rhs)

    file_xdmf["wss_weak_CG_1_cp"].write_checkpoint(
        traction_weak_tan, "wss", t, append=append
    )

    if elem == "th":
        rhs_P2 = df.assemble(df.action(WFv, g_test_P2))
        traction_weak_tan_P2 = df.Function(V_P2, name="wss_weak")
        solver_wss_P2.solve(traction_weak_tan_P2.vector(), rhs_P2)

        file_xdmf["wss_weak_CG_2_cp"].write_checkpoint(
            traction_weak_tan_P2, "wss", t, append=append
        )

    # ==================================
    # Project traction onto the desired function space

    wss_spaces = [("CG", 1)]
    # We compute the standard WSS in the same CG1 space. The DG0 and DG1 projections might be included to compare with commonly used FE spaces for WSS in the literature.
    # wss_spaces = [("DG", 0), ("DG", 1), ("CG", 1)]

    for wss_family, wss_degree in wss_spaces:
        traction_element = df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree)
        traction_space = df.FunctionSpace(mesh, traction_element)
        traction_computer = TractionComputation(traction_space, mark_wall=mark_wall)

        # standard evaluation
        Tn = T(p, v) * n
        wss = traction_computer.project_wss(Tn, n)

        file_xdmf[f"wss_standard_{wss_family}_{wss_degree}_cp"].write_checkpoint(
            wss, "wss", t, append=append
        )
        wss.rename("wss", "wss")
        file_xdmf[f"wss_standard_{wss_family}_{wss_degree}"].write(wss, t)

    return True, t


written_timesteps = []


def evaluate_and_register(i: int):
    """
    Evaluate timestep i and store mapping between original timestep index and
    compact XDMF checkpoint index.

    This is needed because skipped timesteps are not written. Therefore the
    checkpoint index is not necessarily equal to the original timestep index.
    """
    append = len(written_timesteps) > 0
    wrote, t = evaluate_wss_timestep(i, append=append)

    if wrote:
        checkpoint_idx = len(written_timesteps)
        written_timesteps.append((checkpoint_idx, i, t))

        if rank == 0:
            print(
                f"Wrote timestep {i}, t = {t:.6g}, checkpoint index = {checkpoint_idx}"
            )


if stationary:
    evaluate_and_register(ntimesteps - 1)
else:
    for i in range(ntimesteps):
        evaluate_and_register(i)

if rank == 0:
    print(f"Number of written timesteps: {len(written_timesteps)}")


if args.compute_differences:
    if not os.path.exists(f"wss_results/{case}_SI_corrections/csv_files"):
        if rank == 0:
            os.makedirs(
                f"wss_results/{case}_SI_corrections/csv_files/stationary",
                exist_ok=True,
            )
            os.makedirs(
                f"wss_results/{case}_SI/csv_files/pulsatile",
                exist_ok=True,
            )
        comm.Barrier()

    edgelen = re.findall(r"(\d+)um", mesh_file)[0]

    if len(written_timesteps) == 0:
        if rank == 0:
            print(
                f"No timesteps were written because all timestamps are below "
                f"t_begin = {t_begin:.6g}. Skipping difference computation."
            )
    else:
        for checkpoint_i, i, t in written_timesteps:
            # Handle the rescaled velocity case
            if args.theta != 1.0 and args.theta != -1.0:
                wss_rescaled = df.Function(V)

                file_xdmf["wss_rescaled_v_tan_cp"].read_checkpoint(
                    wss_rescaled, "wss", checkpoint_i
                )

                wss_rescaled_norm = compute_norm(wss_rescaled)
                wss_rescaled_max = wss_rescaled_norm.vector().max()

                if rank == 0:
                    print(f"--- Timestep {i}, t = {t:.6g} ---")
                    print(f"{wss_rescaled_max=}\n")

                    if stationary:
                        name_csv = (
                            f"wss_results/{case}_SI_corrections/csv_files/stationary/"
                            f"max_wss_rescaled_{elem}_theta_{args.theta}_stationary.csv"
                        )
                    else:
                        name_csv = (
                            f"wss_results/{case}_SI/csv_files/pulsatile/"
                            f"max_wss_rescaled_{elem}_theta_{args.theta}_pulsatile_{edgelen}.csv"
                        )

                    if not os.path.exists(name_csv):
                        with open(name_csv, "w") as f:
                            if not stationary:
                                f.write("timestep,")
                            f.write("edgelen,max_wss_rescaled\n")

                    with open(name_csv, "a") as f:
                        if not stationary:
                            f.write(f"{i},")
                        f.write(f"{float(edgelen) * 1e-3},{wss_rescaled_max:.2f}\n")

            # Handle the standard/weak evaluation case
            else:
                # CG1
                wss_weak_CG1 = df.Function(
                    df.FunctionSpace(
                        mesh, df.VectorElement("CG", mesh.ufl_cell(), 1)
                    )
                )
                file_xdmf["wss_weak_CG_1_cp"].read_checkpoint(
                    wss_weak_CG1, "wss", checkpoint_i
                )
                wss_weak_CG1_norm = compute_norm(wss_weak_CG1)
                wss_weak_CG1_max = wss_weak_CG1_norm.vector().max()

                if elem == "th":
                    # CG2
                    wss_weak_CG2 = df.Function(
                        df.FunctionSpace(
                            mesh, df.VectorElement("CG", mesh.ufl_cell(), 2)
                        )
                    )
                    file_xdmf["wss_weak_CG_2_cp"].read_checkpoint(
                        wss_weak_CG2, "wss", checkpoint_i
                    )
                    wss_weak_CG2_norm = compute_norm(wss_weak_CG2)
                    wss_weak_CG2_max = wss_weak_CG2_norm.vector().max()
                    diff_weak = (
                        (wss_weak_CG2_max - wss_weak_CG1_max)
                        / wss_weak_CG1_max
                        * 100
                    )

                # DG0
                wss_DG0 = df.Function(
                    df.FunctionSpace(
                        mesh, df.VectorElement("DG", mesh.ufl_cell(), 0)
                    )
                )
                file_xdmf["wss_standard_DG_0_cp"].read_checkpoint(
                    wss_DG0, "wss", checkpoint_i
                )
                wss_DG0_norm = compute_norm(wss_DG0)
                wss_DG0_max = wss_DG0_norm.vector().max()

                # DG1
                wss_DG1 = df.Function(
                    df.FunctionSpace(
                        mesh, df.VectorElement("DG", mesh.ufl_cell(), 1)
                    )
                )
                file_xdmf["wss_standard_DG_1_cp"].read_checkpoint(
                    wss_DG1, "wss", checkpoint_i
                )
                wss_DG1_norm = compute_norm(wss_DG1)
                wss_DG1_max = wss_DG1_norm.vector().max()

                # CG1
                wss_CG1 = df.Function(
                    df.FunctionSpace(
                        mesh, df.VectorElement("CG", mesh.ufl_cell(), 1)
                    )
                )
                file_xdmf["wss_standard_CG_1_cp"].read_checkpoint(
                    wss_CG1, "wss", checkpoint_i
                )
                wss_CG1_norm = compute_norm(wss_CG1)
                wss_CG1_max = wss_CG1_norm.vector().max()

                # relative percentage differences
                diff_DG0 = (wss_DG0_max - wss_CG1_max) / wss_CG1_max * 100
                diff_DG1 = (wss_DG1_max - wss_CG1_max) / wss_CG1_max * 100
                diff_weak_standard = (
                    (wss_weak_CG1_max - wss_CG1_max) / wss_CG1_max * 100
                )

                if rank == 0:
                    print(f"--- Timestep {i}, t = {t:.6g} ---")
                    print(f"{wss_weak_CG1_max=}")
                    if elem == "th":
                        print(f"{wss_weak_CG2_max=}")
                        print(f"{diff_weak=}")
                    print(f"{wss_DG0_max=}")
                    print(f"{wss_DG1_max=}")
                    print(f"{wss_CG1_max=}")
                    print(f"{diff_DG0=}")
                    print(f"{diff_DG1=}")
                    print(f"{diff_weak_standard=}")
                    print("\n")

                    if stationary:
                        name_csv = (
                            f"csv_files/stationary/"
                            f"max_wss_{elem}_theta_{args.theta}_stationary.csv"
                        )
                    else:
                        name_csv = (
                            f"csv_files/pulsatile/"
                            f"max_wss_{elem}_theta_{args.theta}_pulsatile_{edgelen}.csv"
                        )

                    if not os.path.exists(name_csv):
                        with open(name_csv, "w") as f:
                            if not stationary:
                                f.write("timestep,")
                            f.write("edgelen,")
                            f.write("weak_CG1,")
                            f.write("standard_DG0,")
                            f.write("standard_DG1,")
                            f.write("standard_CG1,")
                            f.write("diff_standard_DG0,")
                            f.write("diff_standard_DG1,")
                            f.write("diff_weak_standard")
                            f.write("\n")

                    with open(name_csv, "a") as f:
                        if not stationary:
                            f.write(f"{i},")
                        f.write(f"{float(edgelen) * 1e-3},")
                        f.write(f"{wss_weak_CG1_max:.2f},")
                        f.write(f"{wss_DG0_max:.2f},")
                        f.write(f"{wss_DG1_max:.2f},")
                        f.write(f"{wss_CG1_max:.2f},")
                        f.write(f"{diff_DG0:.2f},")
                        f.write(f"{diff_DG1:.2f},")
                        f.write(f"{diff_weak_standard:.2f}")
                        f.write("\n")