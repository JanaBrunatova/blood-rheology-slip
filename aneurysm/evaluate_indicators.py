import argparse
import xml.etree.ElementTree as ET

import dolfin as df
import numpy as np


def parse_edgelengths(edgelengths_str):
    return [int(x) for x in edgelengths_str.split(",")]


desc = "Compute minimum, maximum, average WSS, LSA, OSI, OVI, and TAWSS for challenge case01 or 02."
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=desc
)
parser.add_argument("--case", default="case01", type=str, choices=["case01", "case02"], dest="case")
parser.add_argument("--mesh-folder", default="", type=str, dest="mesh_folder")
parser.add_argument("--res-folder", default="", type=str, dest="res_folder")
parser.add_argument("--element", default="p1p1", type=str, choices=["p1p1", "th"], dest="element")
parser.add_argument(
    "--edgelengths",
    default="300,250,200,150,100",
    type=parse_edgelengths,
    dest="edgelengths",
)
parser.add_argument("--rescaled", action="store_true", default=False, dest="rescaled")

args = parser.parse_args()
case = args.case
element = args.element
edgelengths = args.edgelengths
rescaled = args.rescaled

print(f"{case=}")
print(f"{element=}")
print(f"{edgelengths=}")
print(f"{rescaled=}")

marker_dome = 42
marker_parent_artery = 73


# ------------------------------------------------------------------ #
#  Time-stepping helpers
# ------------------------------------------------------------------ #

def get_xdmf_times(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    times = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "Time":
            value = elem.attrib.get("Value")
            if value is not None:
                times.append(float(value))

    if len(times) == 0:
        raise RuntimeError(f"No time values found in {filename}")

    times = np.asarray(times, dtype=float)
    print(times)
    if np.any(np.diff(times) <= 0.0):
        raise RuntimeError(f"Time values in {filename} are not strictly increasing")

    return times


def get_trapezoidal_weights(times):
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        raise RuntimeError("At least two time points are needed for time integration")

    dt = np.diff(times)
    weights = np.zeros_like(times)
    weights[0] = 0.5 * dt[0]
    weights[-1] = 0.5 * dt[-1]
    if len(times) > 2:
        weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    total_time = times[-1] - times[0]
    return weights, total_time


def is_pulsatile(filename):
    return len(get_xdmf_times(filename)) > 1


# ------------------------------------------------------------------ #
#  Field helpers
# ------------------------------------------------------------------ #

def compute_norm(data, result=None):
    """
    Compute the pointwise Euclidean norm of a vector Function.
    Returns a scalar Function on the collapsed sub-space of component 0.
    If `result` is provided it must live on that same collapsed space.
    """
    V = data.function_space()
    nx_dofs = V.sub(0).dofmap().dofs()
    ny_dofs = V.sub(1).dofmap().dofs()
    nz_dofs = V.sub(2).dofmap().dofs()

    vals = data.vector().get_local()
    # get_local() returns only the DOFs owned by this process; the index
    # arrays from dofmap().dofs() refer to the *global* vector, so we must
    # restrict them to locally owned DOFs.
    owned = np.arange(data.vector().local_range()[0],
                      data.vector().local_range()[1])
    # Build a mapping global→local for the owned range
    g2l = {g: l for l, g in enumerate(owned)}

    def local_vals(global_dofs):
        return vals[[g2l[g] for g in global_dofs if g in g2l]]

    nx = local_vals(nx_dofs)
    ny = local_vals(ny_dofs)
    nz = local_vals(nz_dofs)
    norm_arr = np.sqrt(nx * nx + ny * ny + nz * nz)

    if result is None:
        S = V.sub(0).collapse()
        result = df.Function(S)

    result.vector().set_local(norm_arr)
    result.vector().apply("insert")
    return result


# ------------------------------------------------------------------ #
#  TAWSS  (trapezoidal time integration of |WSS|)
# ------------------------------------------------------------------ #

def compute_TAWSS(mesh, res_folder, filename, wss_family, wss_degree, output_name="TAWSS"):
    """
    Compute TAWSS = (1/T) * ∫_0^T |τ_w(t)| dt  using trapezoidal rule.
    Returns a scalar CG/DG Function on the mesh.
    """
    V = df.FunctionSpace(mesh, df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree))
    S = df.FunctionSpace(mesh, df.FiniteElement(wss_family, mesh.ufl_cell(), wss_degree))

    wss = df.Function(V)
    wss_mag = df.Function(S)       # reused scratch space each step
    tawss_integral = df.Function(S)  # accumulates ∫|τ| dt

    times = get_xdmf_times(filename)
    weights, total_time = get_trapezoidal_weights(times)

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"Computing TAWSS from {filename}.")
        print(f"  Steps: {len(times)}, interval: [{times[0]:.6e}, {times[-1]:.6e}] s")

    with df.XDMFFile(df.MPI.comm_world, filename) as file_xdmf:
        file_xdmf.parameters["flush_output"] = True
        file_xdmf.parameters["rewrite_function_mesh"] = False

        for i in range(len(times)):
            file_xdmf.read_checkpoint(wss, "wss", i)
            compute_norm(wss, result=wss_mag)                    # |τ_w(t_i)|
            tawss_integral.vector().axpy(weights[i], wss_mag.vector())  # += w_i * |τ|

    # Divide by total time to get the time average
    tawss_integral.vector()[:] /= total_time
    tawss_integral.vector().apply("insert")
    tawss_integral.rename("TAWSS", "TAWSS")

    with df.XDMFFile(df.MPI.comm_world, f"{res_folder}/{output_name}.xdmf") as f:
        f.write(tawss_integral, 0)

    return tawss_integral


# ------------------------------------------------------------------ #
#  OSI  (trapezoidal time integration — both numerator and denominator)
# ------------------------------------------------------------------ #

def compute_OSI(mesh, res_folder, filename, wss_family, wss_degree):
    """
    OSI = 0.5 * (1 - |∫τ_w dt| / ∫|τ_w| dt)

    Both integrals use the trapezoidal rule so non-uniform time steps
    are handled correctly.
    """
    V = df.FunctionSpace(mesh, df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree))
    S = df.FunctionSpace(mesh, df.FiniteElement(wss_family, mesh.ufl_cell(), wss_degree))

    wss = df.Function(V)
    wss_mag = df.Function(S)

    # ∫ τ_w dt  (vector, keeps direction information)
    vec_integral = df.Function(V)
    # ∫ |τ_w| dt  (scalar)
    mag_integral = df.Function(S)

    times = get_xdmf_times(filename)
    weights, _ = get_trapezoidal_weights(times)

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"Computing OSI from {filename}.")
        print(f"  Steps: {len(times)}, interval: [{times[0]:.6e}, {times[-1]:.6e}] s")

    with df.XDMFFile(df.MPI.comm_world, filename) as file_xdmf:
        file_xdmf.parameters["flush_output"] = True
        file_xdmf.parameters["rewrite_function_mesh"] = False

        for i in range(len(times)):
            file_xdmf.read_checkpoint(wss, "wss", i)     # raises on failure — intentional
            compute_norm(wss, result=wss_mag)
            vec_integral.vector().axpy(weights[i], wss.vector())    # ∫ τ dt
            mag_integral.vector().axpy(weights[i], wss_mag.vector())  # ∫ |τ| dt

    # |∫ τ dt|
    mag_of_vec_integral = compute_norm(vec_integral)

    denom = mag_integral.vector().get_local()        # ∫|τ| dt
    numer = mag_of_vec_integral.vector().get_local() # |∫τ dt|

    osi_arr = np.zeros_like(denom)
    mask = denom > 1e-12
    osi_arr[mask] = 0.5 * (1.0 - numer[mask] / denom[mask])

    OSI = df.Function(S)
    OSI.vector().set_local(osi_arr)
    OSI.vector().apply("insert")
    OSI.rename("OSI", "OSI")

    with df.XDMFFile(df.MPI.comm_world, f"{res_folder}/OSI.xdmf") as osi_file:
        osi_file.write_checkpoint(OSI, "OSI", 0)

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"  OSI max: {osi_arr.max():.4f}, mean: {osi_arr.mean():.4f}")

    return OSI


# ------------------------------------------------------------------ #
#  OVI  (same structure as OSI, applied to velocity)
# ------------------------------------------------------------------ #

def compute_OVI(mesh, res_folder, filename, v_family, v_degree):
    """
    OVI = 0.5 * (1 - |∫v dt| / ∫|v| dt)  — trapezoidal rule.
    """
    V = df.FunctionSpace(mesh, df.VectorElement(v_family, mesh.ufl_cell(), v_degree))
    S = df.FunctionSpace(mesh, df.FiniteElement(v_family, mesh.ufl_cell(), v_degree))

    v_fun = df.Function(V)
    v_mag = df.Function(S)
    vec_integral = df.Function(V)
    mag_integral = df.Function(S)

    times = get_xdmf_times(filename)
    weights, _ = get_trapezoidal_weights(times)

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"Computing OVI from {filename}.")
        print(f"  Steps: {len(times)}, interval: [{times[0]:.6e}, {times[-1]:.6e}] s")

    with df.XDMFFile(df.MPI.comm_world, filename) as file_xdmf:
        file_xdmf.parameters["flush_output"] = True
        file_xdmf.parameters["rewrite_function_mesh"] = False

        for i in range(len(times)):
            file_xdmf.read_checkpoint(v_fun, "velocity", i)
            compute_norm(v_fun, result=v_mag)
            vec_integral.vector().axpy(weights[i], v_fun.vector())
            mag_integral.vector().axpy(weights[i], v_mag.vector())

    mag_of_vec_integral = compute_norm(vec_integral)

    denom = mag_integral.vector().get_local()
    numer = mag_of_vec_integral.vector().get_local()

    ovi_arr = np.zeros_like(denom)
    mask = denom > 1e-12
    ovi_arr[mask] = 0.5 * (1.0 - numer[mask] / denom[mask])

    OVI = df.Function(S)
    OVI.vector().set_local(ovi_arr)
    OVI.vector().apply("insert")
    OVI.rename("OVI", "OVI")

    with df.XDMFFile(df.MPI.comm_world, f"{res_folder}/OVI.xdmf") as ovi_file:
        ovi_file.write_checkpoint(OVI, "OVI", 0)

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"  OVI max: {ovi_arr.max():.4f}, mean: {ovi_arr.mean():.4f}")

    return OVI


# ------------------------------------------------------------------ #
#  LSA  — uses the TAWSS scalar field (or |WSS|@t0 for stationary)
# ------------------------------------------------------------------ #

def compute_LSA(res_folder, ds, wss_scalar_fun, parent_avg, marker_dome):
    """
    LSA = (area where TAWSS < 0.1 * parent_avg) / (total dome area) * 100 %

    wss_scalar_fun must be a scalar Function.
    LSA_fun is created in the same FunctionSpace to guarantee DOF alignment.
    """
    low_shear = 0.1 * parent_avg

    # Reuse the exact same FunctionSpace — this is the only safe way to do
    # a DOF-level copy in parallel.
    S = wss_scalar_fun.function_space()
    LSA_fun = df.Function(S)

    wss_arr = wss_scalar_fun.vector().get_local()
    lsa_arr = np.where(wss_arr > low_shear, 0.0, 1.0)
    LSA_fun.vector().set_local(lsa_arr)
    LSA_fun.vector().apply("insert")
    LSA_fun.rename("LSA", "LSA")

    with df.XDMFFile(df.MPI.comm_world, f"{res_folder}/LSA.xdmf") as lsa_file:
        lsa_file.write_checkpoint(LSA_fun, "LSA", 0)

    int_LSA    = df.assemble(LSA_fun * ds(marker_dome))
    total_area = df.assemble(df.Constant(1.0) * ds(marker_dome))
    return LSA_fun, int_LSA / total_area * 100


# ------------------------------------------------------------------ #
#  nWSS  — normalised WSS field
# ------------------------------------------------------------------ #

def compute_nwss(res_folder, wss_scalar_fun, parent_avg):
    """
    nWSS = TAWSS / parent_avg.
    Created in the same FunctionSpace as wss_scalar_fun.
    """
    S = wss_scalar_fun.function_space()
    nwss_fun = df.Function(S)

    arr = wss_scalar_fun.vector().get_local() / parent_avg
    nwss_fun.vector().set_local(arr)
    nwss_fun.vector().apply("insert")
    nwss_fun.rename("nwss", "nwss")

    with df.XDMFFile(df.MPI.comm_world, f"{res_folder}/nwss.xdmf") as f:
        f.write_checkpoint(nwss_fun, "nwss", 0)

    return nwss_fun


# ------------------------------------------------------------------ #
#  Spatial average of a scalar Function over a surface subdomain
# ------------------------------------------------------------------ #

def integrate_scalar_on_subdomain(scalar_fun, ds, marker_id):
    """Area-weighted spatial mean of scalar_fun over the marked boundary patch."""
    total = df.assemble(scalar_fun * ds(marker_id))
    area  = df.assemble(df.Constant(1.0) * ds(marker_id))
    if area <= df.DOLFIN_EPS:
        raise RuntimeError(f"Zero area for boundary marker {marker_id}.")
    return total / area


# ------------------------------------------------------------------ #
#  Min / max / avg of a scalar Function on a surface or volume subdomain
#
#  Vertex/point evaluation in a loop is unreliable in parallel (it
#  triggers a global search) and very slow.  Instead we filter the
#  locally-owned DOFs that sit on the marked entities.
# ------------------------------------------------------------------ #

def evaluate_scalar_on_subdomain(scalar_fun, facet_marker, ds, marker_id):
    """
    Returns (min, max, area-avg) of scalar_fun restricted to the facets
    whose facet_marker == marker_id.

    Min and max are derived from DOF values on owned vertices that belong
    to at least one marked facet — this is correct for CG spaces where
    DOFs live on vertices/edges.
    """
    mesh = scalar_fun.function_space().mesh()
    S    = scalar_fun.function_space()

    # Collect DOFs that touch a marked facet
    dofmap = S.dofmap()
    marked_dofs = set()
    for facet in df.facets(mesh):
        if facet_marker[facet.index()] == marker_id:
            # Get cells adjacent to this facet
            for cell_idx in facet.entities(mesh.topology().dim()):
                cell = df.Cell(mesh, int(cell_idx))
                for dof in dofmap.cell_dofs(cell.index()):
                    marked_dofs.add(int(dof))

    local_range = scalar_fun.vector().local_range()
    vals = scalar_fun.vector().get_local()

    local_marked = [d - local_range[0] for d in marked_dofs
                    if local_range[0] <= d < local_range[1]]

    if local_marked:
        sub_vals = vals[local_marked]
        local_min = float(sub_vals.min())
        local_max = float(sub_vals.max())
    else:
        local_min = float("inf")
        local_max = float("-inf")

    # Reduce across MPI ranks
    comm = df.MPI.comm_world
    global_min = df.MPI.min(comm, local_min)
    global_max = df.MPI.max(comm, local_max)

    total = df.assemble(scalar_fun * ds(marker_id))
    area  = df.assemble(df.Constant(1.0) * ds(marker_id))
    avg   = total / area if area > df.DOLFIN_EPS else 0.0

    return global_min, global_max, avg


def evaluate_scalar_on_volume(scalar_fun, cell_marker, dx_vol, marker_id):
    """
    Returns (min, max, volume-avg) of scalar_fun restricted to cells
    whose cell_marker == marker_id.
    """
    mesh = scalar_fun.function_space().mesh()
    S    = scalar_fun.function_space()
    dofmap = S.dofmap()

    marked_dofs = set()
    for cell in df.cells(mesh):
        if cell_marker[cell.index()] == marker_id:
            for dof in dofmap.cell_dofs(cell.index()):
                marked_dofs.add(int(dof))

    local_range = scalar_fun.vector().local_range()
    vals = scalar_fun.vector().get_local()

    local_marked = [d - local_range[0] for d in marked_dofs
                    if local_range[0] <= d < local_range[1]]

    if local_marked:
        sub_vals = vals[local_marked]
        local_min = float(sub_vals.min())
        local_max = float(sub_vals.max())
    else:
        local_min = float("inf")
        local_max = float("-inf")

    comm = df.MPI.comm_world
    global_min = df.MPI.min(comm, local_min)
    global_max = df.MPI.max(comm, local_max)

    total = df.assemble(scalar_fun * dx_vol(marker_id))
    vol   = df.assemble(df.Constant(1.0) * dx_vol(marker_id))
    avg   = total / vol if vol > df.DOLFIN_EPS else 0.0

    return global_min, global_max, avg


# ------------------------------------------------------------------ #
#  Representative WSS field
# ------------------------------------------------------------------ #

def get_representative_wss(mesh, res_folder, filename, wss_family, wss_degree,
                            output_name="TAWSS"):
    """
    Return (wss_scalar_fun, label) where wss_scalar_fun is:
      - TAWSS field  for pulsatile flow  (multiple timesteps)
      - |WSS| at t=0 for stationary flow (single timestep)
    """
    times = get_xdmf_times(filename)
    pulsatile = len(times) > 1

    if pulsatile:
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print(f"  Pulsatile flow detected ({len(times)} timesteps). "
                  f"Computing TAWSS for spatial averaging.")
        wss_scalar = compute_TAWSS(mesh, res_folder, filename, wss_family, wss_degree,
                                   output_name)
        label = "TAWSS"
    else:
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print("  Stationary flow detected (1 timestep). "
                  "Using |WSS| at t=0 for spatial averaging.")
        V = df.FunctionSpace(mesh, df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree))
        wss_vec = df.Function(V)
        # Use context manager so the file is closed properly
        with df.XDMFFile(df.MPI.comm_world, filename) as file_xdmf:
            file_xdmf.parameters["flush_output"] = True
            file_xdmf.parameters["rewrite_function_mesh"] = False
            file_xdmf.read_checkpoint(wss_vec, "wss", 0)
        wss_scalar = compute_norm(wss_vec)
        label = "WSS (stationary)"

    return wss_scalar, label


# ================================================================== #
#  MAIN LOOP OVER EDGE LENGTHS
# ================================================================== #

for edgelength in edgelengths:

    print(f"\n{'='*60}")
    print(f"edge length = {edgelength * 1e-3} mm")
    print(f"{'='*60}")

    mesh_folder = args.mesh_folder
    res_folder  = args.res_folder

    mesh = df.Mesh()
    with df.HDF5File(df.MPI.comm_world, mesh_folder, "r") as hdf:
        hdf.read(mesh, "/mesh", False)
        dim = mesh.geometry().dim()
        boundary_parts = df.MeshFunction("size_t", mesh, dim - 1, 0)
        hdf.read(boundary_parts, "/boundaries")
    mesh.init()

    facet_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    cell_marker  = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

    if case == "case01":
        origin = [0.0002, -0.00105, 0.00113]
        normal = [0.105, -0.909, -0.403]

        origin_parent_in  = [-0.0043,  0.0113,  0.0076]
        origin_parent_out = [-0.00024, 0.0032,  0.0037]
        normal_parent     = [0.397, -0.826, -0.401]

        for f in df.facets(mesh, "all"):
            if f.exterior():
                facet_marker[f] = 0
                s = f.midpoint()
                dist_dome = (
                    (s[0] - origin[0]) * normal[0]
                    + (s[1] - origin[1]) * normal[1]
                    + (s[2] - origin[2]) * normal[2]
                )
                if dist_dome >= 0:
                    facet_marker[f] = marker_dome

                dist_parent_out = (
                    (s[0] - origin_parent_out[0]) * normal_parent[0]
                    + (s[1] - origin_parent_out[1]) * normal_parent[1]
                    + (s[2] - origin_parent_out[2]) * normal_parent[2]
                )
                dist_parent_in = (
                    (s[0] - origin_parent_in[0]) * normal_parent[0]
                    + (s[1] - origin_parent_in[1]) * normal_parent[1]
                    + (s[2] - origin_parent_in[2]) * normal_parent[2]
                )
                if dist_parent_out <= 0 and dist_parent_in >= 0:
                    facet_marker[f] = marker_parent_artery

        for c in df.cells(mesh):
            s = c.midpoint()
            dist_dome = (
                (s[0] - origin[0]) * normal[0]
                + (s[1] - origin[1]) * normal[1]
                + (s[2] - origin[2]) * normal[2]
            )
            if dist_dome >= 0:
                cell_marker[c] = marker_dome

    elif case == "case02":
        origin1 = [0.00321, -0.00139, -0.00589]
        normal1 = [0.135, -0.429, -0.893]
        origin2 = [0.005, -0.003, -0.002]
        normal2 = [0.5, -0.6, 0.6]

        origin_parent_in   = [-0.0089,  0.00122, -0.00108]
        origin_parent_out  = [-0.0021,  0.0041,  -0.0065]
        normal_parent_in   = [0.74,  0.323, -0.589]
        normal_parent_out  = [0.715, 0.101, -0.69]

        for f in df.facets(mesh, "all"):
            if f.exterior():
                facet_marker[f] = 0
                s = f.midpoint()
                d1 = (
                    (s[0] - origin1[0]) * normal1[0]
                    + (s[1] - origin1[1]) * normal1[1]
                    + (s[2] - origin1[2]) * normal1[2]
                )
                d2 = (
                    (s[0] - origin2[0]) * normal2[0]
                    + (s[1] - origin2[1]) * normal2[1]
                    + (s[2] - origin2[2]) * normal2[2]
                )
                if d1 >= 0 and d2 <= 0:
                    facet_marker[f] = marker_dome

                dist_parent_out = (
                    (s[0] - origin_parent_out[0]) * normal_parent_out[0]
                    + (s[1] - origin_parent_out[1]) * normal_parent_out[1]
                    + (s[2] - origin_parent_out[2]) * normal_parent_out[2]
                )
                dist_parent_in = (
                    (s[0] - origin_parent_in[0]) * normal_parent_in[0]
                    + (s[1] - origin_parent_in[1]) * normal_parent_in[1]
                    + (s[2] - origin_parent_in[2]) * normal_parent_in[2]
                )
                if dist_parent_out <= 0 and dist_parent_in >= 0:
                    facet_marker[f] = marker_parent_artery

        for c in df.cells(mesh):
            s = c.midpoint()
            d1 = (
                (s[0] - origin1[0]) * normal1[0]
                + (s[1] - origin1[1]) * normal1[1]
                + (s[2] - origin1[2]) * normal1[2]
            )
            d2 = (
                (s[0] - origin2[0]) * normal2[0]
                + (s[1] - origin2[1]) * normal2[1]
                + (s[2] - origin2[2]) * normal2[2]
            )
            if d1 >= 0 and d2 <= 0:
                cell_marker[c] = marker_dome

    else:
        raise ValueError("Cannot mark the geometry. Choose either case01 or case02")

    dx     = df.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})
    ds     = df.Measure("ds", domain=mesh, subdomain_data=facet_marker,
                        metadata={"quadrature_degree": 4})
    dx_vol = df.Measure("dx", domain=mesh, subdomain_data=cell_marker,
                        metadata={"quadrature_degree": 4})

    # Sanity-check areas
    area_dome   = df.assemble(df.Constant(1.0) * ds(marker_dome))
    area_parent = df.assemble(df.Constant(1.0) * ds(marker_parent_artery))
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"Dome area:          {area_dome:.6e} m^2")
        print(f"Parent artery area: {area_parent:.6e} m^2")
    if area_dome <= df.DOLFIN_EPS:
        raise RuntimeError("Dome area is zero — check cutting planes.")
    if area_parent <= df.DOLFIN_EPS:
        raise RuntimeError("Parent artery area is zero — check cutting planes.")

    # ==================================================================== #
    #  MAIN EVALUATION BRANCHES
    # ==================================================================== #

    if rescaled:
        print("\n========================")
        print("RESCALED VELOCITY EVALUATION")
        print("========================")
        wss_family = "CG"
        wss_degree = 2 if element == "th" else 1
        filename   = f"{res_folder}/wss_rescaled_v_tan_cp.xdmf"

        # Representative WSS field (TAWSS if pulsatile, |WSS|@t0 otherwise)
        wss_rep, rep_label = get_representative_wss(
            mesh, res_folder, filename, wss_family, wss_degree, output_name="TAWSS"
        )

        # Spatial averages on dome and parent artery
        average_wss_dome   = integrate_scalar_on_subdomain(wss_rep, ds, marker_dome)
        average_wss_parent = integrate_scalar_on_subdomain(wss_rep, ds, marker_parent_artery)

        compute_nwss(res_folder, wss_rep, average_wss_parent)
        _, LSA_rescaled = compute_LSA(res_folder, ds, wss_rep, average_wss_parent, marker_dome)

        # TAWSS statistics on dome
        # wss_rep IS the TAWSS function when pulsatile; reuse it directly.
        if not is_pulsatile(filename):
            tawss_func = compute_TAWSS(
                mesh, res_folder, filename, wss_family, wss_degree, output_name="TAWSS"
            )
        else:
            tawss_func = wss_rep

        min_tawss, max_tawss, avg_tawss = evaluate_scalar_on_subdomain(
            tawss_func, facet_marker, ds, marker_dome
        )

        osi_func = compute_OSI(mesh, res_folder, filename, wss_family, wss_degree)
        min_osi, max_osi, avg_osi = evaluate_scalar_on_subdomain(
            osi_func, facet_marker, ds, marker_dome
        )

        # v_filename = f"{res_folder}/v_cp.xdmf"
        # v_degree   = 2 if element == "th" else 1
        # ovi_func   = compute_OVI(mesh, res_folder, v_filename, "CG", v_degree)
        # min_ovi, max_ovi, avg_ovi = evaluate_scalar_on_volume(
        #     ovi_func, cell_marker, dx_vol, marker_dome
        # )

        print(f"\nRepresentative WSS field:            {rep_label}")
        print(f"average WSS ({rep_label}) dome:          {average_wss_dome:.4f} Pa")
        print(f"average WSS ({rep_label}) parent artery: {average_wss_parent:.4f} Pa")
        print(f"min/max/avg TAWSS rescaled:          {min_tawss:.4f} / {max_tawss:.4f} / {avg_tawss:.4f} Pa")
        print(f"LSA:                                 {LSA_rescaled:.2f} %")
        print(f"min/max/avg OSI:                     {min_osi:.4f} / {max_osi:.4f} / {avg_osi:.4f}")
        # print(f"min/max/avg OVI:                     {min_ovi:.4f} / {max_ovi:.4f} / {avg_ovi:.4f}")

    else:
        # ---------------------------------------------------------------- #
        #  BOUNDARY-FLUX  (weak CG1)
        # ---------------------------------------------------------------- #
        print("\n========================")
        print("BOUNDARY-FLUX EVALUATION")
        print("========================")
        wss_family = "CG"
        wss_degree = 1
        filename   = f"{res_folder}/wss_weak_CG_1_cp.xdmf"

        wss_rep_weak, rep_label_weak = get_representative_wss(
            mesh, res_folder, filename, wss_family, wss_degree,
            output_name="TAWSS_boundary_flux_CG1"
        )
        average_wss_weak_dome   = integrate_scalar_on_subdomain(wss_rep_weak, ds, marker_dome)
        average_wss_weak_parent = integrate_scalar_on_subdomain(wss_rep_weak, ds, marker_parent_artery)
        _, LSA_weak = compute_LSA(
            res_folder, ds, wss_rep_weak, average_wss_weak_parent, marker_dome
        )

        # For pulsatile flow wss_rep_weak is already TAWSS; avoid a second pass.
        if not is_pulsatile(filename):
            tawss_weak_func = compute_TAWSS(
                mesh, res_folder, filename, wss_family, wss_degree,
                output_name="TAWSS_boundary_flux_CG1"
            )
        else:
            tawss_weak_func = wss_rep_weak

        min_tawss_weak, max_tawss_weak, avg_tawss_weak = evaluate_scalar_on_subdomain(
            tawss_weak_func, facet_marker, ds, marker_dome
        )

        print(f"Representative WSS field:   {rep_label_weak}")
        print(f"average WSS dome:           {average_wss_weak_dome:.4f} Pa")
        print(f"average WSS parent artery:  {average_wss_weak_parent:.4f} Pa")
        print(f"min/max/avg TAWSS BF CG1:   {min_tawss_weak:.4f} / {max_tawss_weak:.4f} / {avg_tawss_weak:.4f} Pa")
        print(f"LSA boundary-flux:          {LSA_weak:.2f} %")

        # ---------------------------------------------------------------- #
        #  STANDARD PROJECTIONS
        # ---------------------------------------------------------------- #
        print("\n========================")
        print("P1 (AND OPTIONALLY DG1/DG0) PROJECTION")
        print("========================")

        for proj_family, proj_degree, proj_label, tawss_out in [
            ("CG", 1, "P1", "TAWSS"),
            # ("DG", 1, "DG1", "TAWSS_standard_DG1"),
            # ("DG", 0, "DG0", "TAWSS_standard_DG0"),
        ]:
            print(f"\n--- {proj_label} ---")
            filename = f"{res_folder}/wss_standard_{proj_family}_{proj_degree}_cp.xdmf"

            wss_rep, rep_label = get_representative_wss(
                mesh, res_folder, filename, proj_family, proj_degree, output_name=tawss_out
            )
            avg_dome   = integrate_scalar_on_subdomain(wss_rep, ds, marker_dome)
            avg_parent = integrate_scalar_on_subdomain(wss_rep, ds, marker_parent_artery)
            _, LSA = compute_LSA(res_folder, ds, wss_rep, avg_parent, marker_dome)

            if not is_pulsatile(filename):
                tawss_func = compute_TAWSS(
                    mesh, res_folder, filename, proj_family, proj_degree, output_name=tawss_out
                )
            else:
                tawss_func = wss_rep

            min_t, max_t, avg_t = evaluate_scalar_on_subdomain(
                tawss_func, facet_marker, ds, marker_dome
            )

            print(f"Representative WSS field ({proj_label}): {rep_label}")
            print(f"average WSS {proj_label} dome:           {avg_dome:.4f} Pa")
            print(f"average WSS {proj_label} parent artery:  {avg_parent:.4f} Pa")
            print(f"min/max/avg TAWSS {proj_label}:          {min_t:.4f} / {max_t:.4f} / {avg_t:.4f} Pa")
            print(f"LSA {proj_label}:                        {LSA:.2f} %")

            # OSI, OVI and nWSS only for CG1
            if proj_family == "CG" and proj_degree == 1:
                compute_nwss(res_folder, wss_rep, avg_parent)

                osi_func = compute_OSI(mesh, res_folder, filename, proj_family, proj_degree)
                min_osi, max_osi, avg_osi = evaluate_scalar_on_subdomain(
                    osi_func, facet_marker, ds, marker_dome
                )

                # v_filename = f"{res_folder}/v_cp.xdmf"
                # v_degree   = 2 if element == "th" else 1
                # ovi_func   = compute_OVI(mesh, res_folder, v_filename, "CG", v_degree)
                # min_ovi, max_ovi, avg_ovi = evaluate_scalar_on_volume(
                #     ovi_func, cell_marker, dx_vol, marker_dome
                # )

                print(f"min/max/avg OSI {proj_label}:         {min_osi:.4f} / {max_osi:.4f} / {avg_osi:.4f}")
                # print(f"min/max/avg OVI:                     {min_ovi:.4f} / {max_ovi:.4f} / {avg_ovi:.4f}")