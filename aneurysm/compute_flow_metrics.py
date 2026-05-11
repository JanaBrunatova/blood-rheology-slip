import argparse
import os
import xml.etree.ElementTree as ET

import dolfin as df
import numpy as np

from ns_aneurysm import model


def parse_marker_list(marker_str):
    return [int(x.strip()) for x in marker_str.split(",") if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compute time-averaged bulk, boundary, and total dissipation, "
            "vorticity, inlet-averaged pressure, and TAWSS over the full wall "
            "and aneurysm wall, then save results to a txt file."
        ),
    )
    parser.add_argument(
        "--case",
        default="case01",
        choices=["case01", "case02"],
        help="Challenge case used for aneurysm/domain marking.",
    )
    parser.add_argument(
        "--mesh-folder",
        required=True,
        dest="mesh_folder",
        help="Path to the marked mesh HDF5 file.",
    )
    parser.add_argument(
        "--res-folder",
        required=True,
        dest="res_folder",
        help="Path to the folder with result XDMF files.",
    )
    parser.add_argument(
        "--element",
        default="th",
        choices=["p1p1", "th"],
        help="Finite element used for the velocity solution.",
    )
    parser.add_argument(
        "--mu",
        default=0.00345,
        type=float,
        help="Dynamic viscosity in Pa.s.",
    )
    parser.add_argument(
        "--theta",
        required=True,
        type=float,
        help="Theta value for the Navier-slip boundary dissipation coefficient.",
    )
    parser.add_argument(
        "--wall-markers",
        default="1",
        type=parse_marker_list,
        dest="wall_markers",
        help=(
            "Comma-separated list of boundary_parts markers that belong to the "
            "vessel wall, for example 1 or 1,5."
        ),
    )
    parser.add_argument(
        "--v-file",
        default="v_cp.xdmf",
        dest="velocity_file",
        help="Velocity XDMF file name inside res_folder.",
    )
    parser.add_argument(
        "--p-file",
        default="p_cp.xdmf",
        dest="pressure_file",
        help="Pressure XDMF file name inside res_folder.",
    )
    parser.add_argument(
        "--wss-file",
        default="wss_standard_CG_1_cp.xdmf",
        dest="wss_file",
        help="WSS XDMF file name inside res_folder.",
    )
    parser.add_argument(
        "--velocity-name",
        default="velocity",
        dest="velocity_name",
        help="Checkpoint field name for velocity in the XDMF file.",
    )
    parser.add_argument(
        "--pressure-name",
        default="pressure",
        dest="pressure_name",
        help="Checkpoint field name for pressure in the XDMF file.",
    )
    parser.add_argument(
        "--wss-name",
        default="wss",
        dest="wss_name",
        help="Checkpoint field name for WSS in the XDMF file.",
    )
    parser.add_argument(
        "--wss-family",
        default="CG",
        choices=["CG", "DG"],
        dest="wss_family",
        help="Finite element family used for stored WSS.",
    )
    parser.add_argument(
        "--wss-degree",
        default=1,
        type=int,
        dest="wss_degree",
        help="Finite element degree used for stored WSS.",
    )
    parser.add_argument(
        "--quadrature-degree",
        default=4,
        type=int,
        dest="quadrature_degree",
        help="Quadrature degree for integration.",
    )
    parser.add_argument(
        "--osi-file",
        default="OSI.xdmf",
        dest="osi_file",
        help="OSI XDMF file name inside res_folder.",
    )
    parser.add_argument(
        "--lsa-file",
        default="LSA.xdmf",
        dest="lsa_file",
        help="LSA XDMF file name inside res_folder.",
    )
    parser.add_argument(
        "--nwss-file",
        default="nwss.xdmf",
        dest="nwss_file",
        help="NWSS XDMF file name inside res_folder.",
    )
    return parser


def mark_aneurysm_regions(mesh, case):
    marker_aneurysm = 42

    facet_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    cell_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

    if case == "case01":
        origin = [0.0002, -0.00105, 0.00113]
        normal = [0.105, -0.909, -0.403]

        for f in df.facets(mesh):
            if f.exterior():
                s = f.midpoint()
                distance = (
                    (s[0] - origin[0]) * normal[0]
                    + (s[1] - origin[1]) * normal[1]
                    + (s[2] - origin[2]) * normal[2]
                )
                if distance >= 0.0:
                    facet_marker[f] = marker_aneurysm

        for c in df.cells(mesh):
            s = c.midpoint()
            distance = (
                (s[0] - origin[0]) * normal[0]
                + (s[1] - origin[1]) * normal[1]
                + (s[2] - origin[2]) * normal[2]
            )
            if distance >= 0.0:
                cell_marker[c] = marker_aneurysm

    elif case == "case02":
        origin1 = [0.00321, -0.00139, -0.00589]
        normal1 = [0.135, -0.429, -0.893]

        origin2 = [0.005, -0.003, -0.002]
        normal2 = [0.5, -0.6, 0.6]

        for f in df.facets(mesh):
            if f.exterior():
                s = f.midpoint()
                distance1 = (
                    (s[0] - origin1[0]) * normal1[0]
                    + (s[1] - origin1[1]) * normal1[1]
                    + (s[2] - origin1[2]) * normal1[2]
                )
                distance2 = (
                    (s[0] - origin2[0]) * normal2[0]
                    + (s[1] - origin2[1]) * normal2[1]
                    + (s[2] - origin2[2]) * normal2[2]
                )
                if (distance1 >= 0.0) and (distance2 <= 0.0):
                    facet_marker[f] = marker_aneurysm

        for c in df.cells(mesh):
            s = c.midpoint()
            distance1 = (
                (s[0] - origin1[0]) * normal1[0]
                + (s[1] - origin1[1]) * normal1[1]
                + (s[2] - origin1[2]) * normal1[2]
            )
            distance2 = (
                (s[0] - origin2[0]) * normal2[0]
                + (s[1] - origin2[1]) * normal2[1]
                + (s[2] - origin2[2]) * normal2[2]
            )
            if (distance1 >= 0.0) and (distance2 <= 0.0):
                cell_marker[c] = marker_aneurysm

    else:
        raise ValueError("Cannot mark the geometry. Choose either case01 or case02.")

    return facet_marker, cell_marker, marker_aneurysm


def mark_wall_regions(mesh, boundary_parts, facet_marker, marker_aneurysm, wall_markers):
    wall_full_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    wall_aneurysm_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    wall_marker_set = set(wall_markers)

    for f in df.facets(mesh):
        if not f.exterior():
            continue

        bmark = boundary_parts[f]
        if bmark in wall_marker_set:
            wall_full_marker[f] = 1
            if facet_marker[f] == marker_aneurysm:
                wall_aneurysm_marker[f] = 1

    return wall_full_marker, wall_aneurysm_marker


def average_on_boundary(expr, ds_boundary, marker_id):
    area = df.assemble(df.Constant(1.0) * ds_boundary(marker_id))
    if area <= df.DOLFIN_EPS:
        raise RuntimeError(f"Boundary marker {marker_id} has zero area.")
    return df.assemble(expr * ds_boundary(marker_id)) / area


def get_xdmf_times(xdmf_path):
    tree = ET.parse(xdmf_path)
    root = tree.getroot()

    times = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "Time":
            value = elem.attrib.get("Value")
            if value is not None:
                times.append(float(value))

    if not times:
        raise RuntimeError(f"No time values found in {xdmf_path}.")

    times = np.asarray(times, dtype=float)

    if np.any(np.diff(times) < 0.0):
        raise RuntimeError(f"Non-monotone time values found in {xdmf_path}.")

    return times


def time_average(times, values):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if times.ndim != 1 or values.ndim != 1:
        raise ValueError("times and values must be one-dimensional arrays.")
    if len(times) != len(values):
        raise ValueError("times and values must have the same length.")
    if len(times) < 2:
        raise ValueError("At least two time points are required for time integration.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Time values must be strictly increasing.")

    total_time = times[-1] - times[0]
    if total_time <= 0.0:
        raise ValueError("Total time interval must be positive.")

    return float(np.trapz(values, x=times) / total_time)



def main():
    parser = build_parser()
    args = parser.parse_args()

    inlet_marker = 2
    gamma = 1.0
    # For theta = 1, the term is 0.0 (no-slip/no boundary dissipation)
    if np.isclose(abs(args.theta), 1.0):
        boundary_diss_coeff = 0.0
    else:
        boundary_diss_coeff = 2.0 * np.pi * abs(args.theta) / (gamma * (1.0 - abs(args.theta)))

    mesh = df.Mesh()
    with df.HDF5File(df.MPI.comm_world, args.mesh_folder, "r") as hdf:
        hdf.read(mesh, "/mesh", False)
        dim = mesh.geometry().dim()
        boundary_parts = df.MeshFunction("size_t", mesh, dim - 1, 0)
        hdf.read(boundary_parts, "/boundaries")
    mesh.init()

    facet_marker, cell_marker, marker_aneurysm = mark_aneurysm_regions(mesh, args.case)
    wall_full_marker, wall_aneurysm_marker = mark_wall_regions(
        mesh, boundary_parts, facet_marker, marker_aneurysm, args.wall_markers
    )

    metadata = {"quadrature_degree": args.quadrature_degree}
    dx = df.Measure("dx", domain=mesh, metadata=metadata)
    dx_aneurysm = df.Measure(
        "dx", domain=mesh, subdomain_data=cell_marker, metadata=metadata
    )
    ds_boundary = df.Measure(
        "ds", domain=mesh, subdomain_data=boundary_parts, metadata=metadata
    )
    ds_wall_full = df.Measure(
        "ds", domain=mesh, subdomain_data=wall_full_marker, metadata=metadata
    )
    ds_wall_aneurysm = df.Measure(
        "ds", domain=mesh, subdomain_data=wall_aneurysm_marker, metadata=metadata
    )

    volume_full = df.assemble(df.Constant(1.0) * dx)
    volume_aneurysm = df.assemble(df.Constant(1.0) * dx_aneurysm(marker_aneurysm))
    area_wall_full = df.assemble(df.Constant(1.0) * ds_wall_full(1))
    area_wall_aneurysm = df.assemble(df.Constant(1.0) * ds_wall_aneurysm(1))
    area_inlet = df.assemble(df.Constant(1.0) * ds_boundary(inlet_marker))

    if volume_full <= df.DOLFIN_EPS:
        raise RuntimeError("Full domain volume is zero.")
    if volume_aneurysm <= df.DOLFIN_EPS:
        raise RuntimeError("Aneurysm volume is zero. Check case selection or cutting planes.")
    if area_wall_full <= df.DOLFIN_EPS:
        raise RuntimeError("Full wall area is zero. Check wall markers.")
    if area_wall_aneurysm <= df.DOLFIN_EPS:
        raise RuntimeError("Aneurysm wall area is zero. Check wall markers and aneurysm cut.")
    if area_inlet <= df.DOLFIN_EPS:
        raise RuntimeError("Inlet area is zero for boundary marker 2.")

    v_family = "CG"
    v_degree = 2 if args.element == "th" else 1
    p_family = "CG"
    p_degree = 1

    V = df.FunctionSpace(mesh, df.VectorElement(v_family, mesh.ufl_cell(), v_degree))
    Q = df.FunctionSpace(mesh, df.FiniteElement(p_family, mesh.ufl_cell(), p_degree))
    W = df.FunctionSpace(
        mesh, df.VectorElement(args.wss_family, mesh.ufl_cell(), args.wss_degree)
    )

    velocity = df.Function(V)
    pressure = df.Function(Q)
    wss = df.Function(W)

    n = df.FacetNormal(mesh)

    velocity_path = os.path.join(args.res_folder, args.velocity_file)
    pressure_path = os.path.join(args.res_folder, args.pressure_file)
    wss_path = os.path.join(args.res_folder, args.wss_file)
    
    model_name_str = os.path.basename(os.path.normpath(args.res_folder))
    print(f"{model_name_str=}")
    if model_name_str.startswith("Carreau"):
        model_name = getattr(model, model_name_str)
        visc_model = model_name(mesh, velocity, scale=0.1)
        mu = visc_model.viscosity
    else:
        mu = args.mu    

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"case={args.case}")
        print(f"element={args.element}")
        print(f"theta={args.theta}")
        print(f"gamma={gamma}")
        print(f"mu={mu}")
        print(f"boundary_diss_coeff={boundary_diss_coeff}")
        print(f"wall_markers={args.wall_markers}")
        print(f"Full domain volume: {volume_full:.6e} m^3")
        print(f"Aneurysm volume: {volume_aneurysm:.6e} m^3")
        print(f"Full wall area: {area_wall_full:.6e} m^2")
        print(f"Aneurysm wall area: {area_wall_aneurysm:.6e} m^2")
        print(f"Inlet area (marker 2): {area_inlet:.6e} m^2")
        print(f"Velocity file: {velocity_path}")
        print(f"Pressure file: {pressure_path}")
        print(f"WSS file: {wss_path}")

    velocity_file = df.XDMFFile(df.MPI.comm_world, velocity_path)
    pressure_file = df.XDMFFile(df.MPI.comm_world, pressure_path)
    wss_file = df.XDMFFile(df.MPI.comm_world, wss_path)

    times_all = get_xdmf_times(velocity_path)
    nsteps = len(times_all)
    # nsteps = 10
    times = times_all[:nsteps]

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f"velocity timesteps = {len(times_all)}")
        print(f"using nsteps = {nsteps}")
        if nsteps >= 2:
            print(f"time interval = [{times[0]:.6e}, {times[-1]:.6e}] s")

    records = []

    for step in range(nsteps):
        t = times[step]
        try:
            # print("read v")
            velocity_file.read_checkpoint(velocity, args.velocity_name, step)
            # print("read p")
            pressure_file.read_checkpoint(pressure, args.pressure_name, step)
            # print("read wss")
            wss_file.read_checkpoint(wss, args.wss_name, step)
            # print("read done")
        except Exception as e:
            velocity_file.close()
            pressure_file.close()
            wss_file.close()
            raise RuntimeError(f"Read failed at step {step}, time {t}: {e}")


        # model_name_str = os.path.basename(os.path.normpath(args.res_folder))
        # if model_name_str.startswith("Carreau"):
        #     model_name = getattr(model, model_name_str)
        #     visc_model = model_name(mesh, velocity, scale=0.1)
        #     mu = visc_model.viscosity
        #     visc_model.project_viscosity()
        # else:
        #     mu = args.mu
        if model_name_str.startswith("Carreau"):
            visc_model.project_viscosity()
        
        Dv = df.sym(df.grad(velocity))
        bulk_diss_density = 2.0 * mu * df.inner(Dv, Dv)

        v_t = velocity - df.inner(velocity, n) * n
        boundary_diss_density = boundary_diss_coeff * df.inner(v_t, v_t)

        vorticity_magnitude = df.sqrt(df.inner(df.curl(velocity), df.curl(velocity)))
        wss_magnitude = df.sqrt(df.inner(wss, wss))

        bulk_diss_full = df.assemble(bulk_diss_density * dx)
        bulk_diss_aneurysm = df.assemble(bulk_diss_density * dx_aneurysm(marker_aneurysm))

        boundary_diss_full = df.assemble(boundary_diss_density * ds_wall_full(1))
        boundary_diss_aneurysm = df.assemble(
            boundary_diss_density * ds_wall_aneurysm(1)
        )

        total_diss_full = bulk_diss_full + boundary_diss_full
        total_diss_aneurysm = bulk_diss_aneurysm + boundary_diss_aneurysm

        mean_vorticity_full = df.assemble(vorticity_magnitude * dx) / volume_full
        mean_vorticity_aneurysm = (
            df.assemble(vorticity_magnitude * dx_aneurysm(marker_aneurysm))
            / volume_aneurysm
        )

        mean_pressure_inlet = average_on_boundary(pressure, ds_boundary, inlet_marker)

        mean_wss_full = df.assemble(wss_magnitude * ds_wall_full(1)) / area_wall_full
        mean_wss_aneurysm = (
            df.assemble(wss_magnitude * ds_wall_aneurysm(1)) / area_wall_aneurysm
        )

        records.append(
            [
                float(t),
                float(bulk_diss_full),
                float(bulk_diss_aneurysm),
                float(boundary_diss_full),
                float(boundary_diss_aneurysm),
                float(total_diss_full),
                float(total_diss_aneurysm),
                float(mean_vorticity_full),
                float(mean_vorticity_aneurysm),
                float(mean_pressure_inlet),
                float(mean_wss_full),
                float(mean_wss_aneurysm),
            ]
        )

        if df.MPI.rank(df.MPI.comm_world) == 0:
            print(
                f"step {step}, t={t:.6e}: "
                f"bulk_diss_aneurysm={bulk_diss_aneurysm:.6e}, "
                f"bndry_diss_aneurysm={boundary_diss_aneurysm:.6e}, "
                f"total_diss_aneurysm={total_diss_aneurysm:.6e}, "
                f"vort_aneurysm={mean_vorticity_aneurysm:.6e}, "
                f"pin={mean_pressure_inlet:.6e}, "
                f"tawss_full_inst={mean_wss_full:.6e}, "
                f"tawss_aneurysm_inst={mean_wss_aneurysm:.6e}"
            )

    velocity_file.close()
    pressure_file.close()
    wss_file.close()

    if not records:
        raise RuntimeError(
            "No checkpoints were read. Check filenames, checkpoint names, and function spaces."
        )

    data = np.asarray(records, dtype=float)

    times_used = data[:, 0]
    bulk_diss_aneurysm_values = data[:, 2]
    boundary_diss_aneurysm_values = data[:, 4]
    total_diss_aneurysm_values = data[:, 6]
    mean_vorticity_aneurysm_values = data[:, 8]
    mean_pressure_inlet_values = data[:, 9]
    mean_wss_aneurysm_values = data[:, 11]

    time_avg_bulk_diss_aneurysm = time_average(times_used, bulk_diss_aneurysm_values)
    time_avg_boundary_diss_aneurysm = time_average(
        times_used, boundary_diss_aneurysm_values
    )
    time_avg_total_diss_aneurysm = time_average(times_used, total_diss_aneurysm_values)
    time_avg_mean_vorticity_aneurysm = time_average(
        times_used, mean_vorticity_aneurysm_values
    )
    time_avg_mean_pressure_inlet = time_average(times_used, mean_pressure_inlet_values)
    tawss_aneurysm = time_average(times_used, mean_wss_aneurysm_values)
    
    # Define a scalar function space for OSI, LSA, and NWSS
    # Using CG 1 as standard, adjust if your XDMF was written with DG or higher order
    if abs(args.theta)<1:
        degree = 2
    else:
        degree = 1
    S = df.FunctionSpace(mesh, df.FiniteElement("CG", mesh.ufl_cell(), degree))
    
    osi_func = df.Function(S)
    lsa_func = df.Function(S)
    nwss_func = df.Function(S)

    osi_path = os.path.join(args.res_folder, args.osi_file)
    lsa_path = os.path.join(args.res_folder, args.lsa_file)
    nwss_path = os.path.join(args.res_folder, args.nwss_file)

    # Read the static fields directly
    if os.path.isfile(osi_path):
        with df.XDMFFile(df.MPI.comm_world, osi_path) as f_osi:
            f_osi.read_checkpoint(osi_func, "OSI", 0)
        # Compute spatial average over the aneurysm wall
        mean_osi_aneurysm = df.assemble(osi_func * ds_wall_aneurysm(1)) / area_wall_aneurysm
    else:
        mean_osi_aneurysm = 0.0
    
    if os.path.isfile(lsa_path):
        with df.XDMFFile(df.MPI.comm_world, lsa_path) as f_lsa:
            f_lsa.read_checkpoint(lsa_func, "LSA", 0)
        mean_lsa_aneurysm = df.assemble(lsa_func * ds_wall_aneurysm(1)) / area_wall_aneurysm
    else:
        mean_lsa_aneurysm = 0.0

    if os.path.isfile(nwss_path):
        with df.XDMFFile(df.MPI.comm_world, nwss_path) as f_nwss:
            f_nwss.read_checkpoint(nwss_func, "nwss", 0)
        mean_nwss_aneurysm = df.assemble(nwss_func * ds_wall_aneurysm(1)) / area_wall_aneurysm
    else:
        mean_nwss_aneurysm = 0.0
    

    # Ensure the path is normalized to handle trailing slashes correctly
    normalized_path = os.path.normpath(args.res_folder)
    
    # Extract the last component of the path
    model_name = os.path.basename(normalized_path)

    if df.MPI.rank(df.MPI.comm_world) == 0:
        model_suffix = model_name
        
        # Ensure the directory exists
        output_dir = "output_tables/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # ------------------------------------------------------------------
        # 1. Time-averaged summary CSV (one row per run, appended)
        # ------------------------------------------------------------------
        csv_path = os.path.join(output_dir, f"averaged_hemodynamics_{model_suffix}.csv")
        
        # Use generic headers (the LaTeX code handles the N/C mapping)
        header = "theta,kappa,D_bulk,D_bnd,D_tot,Pin,WSS,Vort,OSI,LSA,NWSS\n"
        
        file_exists = os.path.isfile(csv_path)

        # kappa = theta / (1 - theta); inf when |theta| = 1
        abs_theta = abs(args.theta)
        if np.isclose(abs_theta, 1.0):
            kappa_str = "$\infty$"
        else:
            kappa_str = f"{abs_theta / (1.0 - abs_theta):.2g}"

        # Pack only the current model's data
        data_row = [
            f"{abs_theta:.2g}",
            kappa_str,
            f"{time_avg_bulk_diss_aneurysm:.6e}",
            f"{time_avg_boundary_diss_aneurysm:.6e}",
            f"{time_avg_total_diss_aneurysm:.6e}",
            f"{time_avg_mean_pressure_inlet:.6e}",
            f"{tawss_aneurysm:.6e}",
            f"{time_avg_mean_vorticity_aneurysm:.6e}",
            f"{mean_osi_aneurysm:.6e}",
            f"{mean_lsa_aneurysm:.6e}",
            f"{mean_nwss_aneurysm:.6e}",
        ]
        
        mode = "a" if file_exists else "w"
        with open(csv_path, mode, encoding="utf-8") as f:
            if not file_exists:
                f.write(header)
            f.write(",".join(data_row) + "\n")
            
        print(f"Results appended to {csv_path}")

        # ------------------------------------------------------------------
        # 2. Temporal CSV (one row per timestep, written fresh each run)
        # ------------------------------------------------------------------
        if abs(args.theta)<1.0:
            theta_str = f"{abs(args.theta)*100:.2g}"
        else:
            theta_str = 1
        temporal_csv_path = os.path.join(
            output_dir,
            f"temporal_hemodynamics_{model_suffix}_theta_{theta_str}.csv",
        )

        temporal_header = (
            "step,time,"
            "bulk_diss_aneurysm,boundary_diss_aneurysm,total_diss_aneurysm,"
            "mean_vorticity_aneurysm,mean_pressure_inlet,mean_wss_aneurysm\n"
        )

        with open(temporal_csv_path, "w", encoding="utf-8") as f:
            f.write(temporal_header)
            for step_idx, (
                t,
                bd_a,
                bnd_a,
                tot_a,
                vort_a,
                pin,
                wss_a,
            ) in enumerate(
                zip(
                    times_used,
                    bulk_diss_aneurysm_values,
                    boundary_diss_aneurysm_values,
                    total_diss_aneurysm_values,
                    mean_vorticity_aneurysm_values,
                    mean_pressure_inlet_values,
                    mean_wss_aneurysm_values,
                )
            ):
                f.write(
                    f"{step_idx},{t:.6e},"
                    f"{bd_a:.6e},{bnd_a:.6e},{tot_a:.6e},"
                    f"{vort_a:.6e},{pin:.6e},{wss_a:.6e}\n"
                )

        print(f"Temporal data written to {temporal_csv_path}")


if __name__ == "__main__":
    main()