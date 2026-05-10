# Aortic Root Hemodynamics Simulations

This folder contains FEM simulations of blood flow in a 2D axisymmetric aortic root geometry using FEniCS.
The code solves either Newtonian or Carreau (non-Newtonian) flow equations to steady state or with time-dependent pulsatile inlet conditions.

## Codes

| Script | Model | Type | Output |
|--------|-------|------|--------|
| `aorta-2d-cyl-NS.py` | Newtonian | Steady-state | Text file |
| `aorta-2d-cyl-Carreau.py` | Carreau rheology | Steady-state | Text file |
| `aorta-2d-cyl-NS-evol.py` | Newtonian | Pulsatile (time-dependent) | Text file + VTK |
| `aorta-2d-cyl-Carreau-evol.py` | Carreau rheology | Pulsatile (time-dependent) | Text file + VTK |

## Mesh Files

Three axisymmetric mesh geometries, parameterized by sinus radius:
- `mesh_R12.h5`: Sinus radius 12 mm
- `mesh_R16.h5`: Sinus radius 16 mm
- `mesh_R20.h5`: Sinus radius 20 mm

The desired mesh is automatically selected via the 4th command-line argument.

## Running Simulations

### Command Format

```
python3 <script_name>.py <kappa> <output_file> <beta> <radius> [<hematocrit>]
```

### Arguments

| Argument | Description | Type | Example |
|----------|-------------|------|---------|
| `kappa` | Navier-slip coefficient (Pa·s/m). Use 0 for perfect slip, large value → no-slip limit | float | 2 |
| `output_file` | Path to output file for scalar hemodynamic quantities | string | `output_NS_45.txt` |
| `beta` | Nitsche penalty parameter for wall boundary enforcement | float | 1000 |
| `radius` | Sinus radius in mm (12, 16, or 20) — selects mesh file automatically | int | 16 |
| `hematocrit` | Blood hematocrit level (optional; choices: 25, 45, 65 %; default: 45) | int | 45 |

### Examples

**Steady-state Newtonian (R=16 mm, $\kappa$=2 Pa·s/m, Hct=45%)**
```bash
python3 aorta-2d-cyl-NS.py 2 output_NS_45.txt 1000 16 45
```

**Steady-state Carreau (R=16 mm, $\kappa$=2 Pa·s/m, Hct=45%)**
```bash
python3 aorta-2d-cyl-Carreau.py 2 output_Carreau_45.txt 1000 16 45
```

**Pulsatile Newtonian (R=12 mm, $\kappa$=3.14 Pa·s/m, Hct=65%)**
```bash
python3 aorta-2d-cyl-NS-evol.py 3.14 output_NS_evol.txt 1000 12 65
```

**Pulsatile Carreau (R=20 mm, $\kappa$=3.14 Pa·s/m, default Hct=45%)**
```bash
python3 aorta-2d-cyl-Carreau-evol.py 3.14 output_Carreau_evol.txt 1000 20
```

## Output

### Text Output
Steady scripts append one final line after convergence; evolution scripts append one line per time step:
```
<time> <kappa> <bulk_dissipation> <boundary_dissipation> <total_dissipation> <wall_shear_stress> <pressure_drop> <vorticity> <normal_velocity>
```

### VTK Output (Evolution Scripts Only)
Velocity and pressure fields are exported to:
- `results_NS_<hematocrit>/` (for NS-evol)
- `results_Carreau_<hematocrit>/` (for Carreau-evol)

Visualize with [ParaView](https://www.paraview.org/):
```bash
paraview results_NS_45/v.xdmf
```

## Physical Model

### Carreau Rheological Model
For non-Newtonian flow:
$$\mu(\dot{\gamma}) = \mu_\infty + (\mu_0 - \mu_\infty) \left[1 + (\lambda \dot{\gamma})^2\right]^{(n-1)/2}$$

Blood viscosity parameters (built-in, by hematocrit):

| Hematocrit | $\mu_0$ (Pa·s) | $\mu_\infty$ (Pa·s) | $\lambda$ (s) | $n$ |
|:----------:|:--------------:|:------------------:|:----:|:-----:|
| 25% | 0.0178 | 0.00257 | 12.448 | 0.330 |
| 45% | 0.1610 | 0.00345 | 39.418 | 0.479 |
| 65% | 0.8592 | 0.00802 | 103.088 | 0.389 |

### Boundary Conditions
- **Inlet (z = -0.022)**: Poiseuille velocity profile with Navier-slip (parameter κ)
- **Outlet (z = 0.022)**: Directional do-nothing condition
- **Wall**: Slip + impermeabilitycondition enforced via Nitsche penalty method (parameter β)
- **Axis (r = 0)**: Symmetry condition

### Boundary Normal Projection
The module `generate_normal_2D.py` projects exterior facet normals onto CG1 space, enabling accurate integration of boundary terms on curved walls (essential for the Nitsche method in complex geometries).

## Numerical Methods

### Discretization
- **Velocity field (v)**: Continuous Galerkin (CG), order 2
- **Pressure field (p)**: Continuous Galerkin (CG), order 1
- **Mesh**: Unstructured triangle elements (generated offline, stored in HDF5)

### Time Integration
- **Steady scripts** (`aorta-2d-cyl-NS.py`, `aorta-2d-cyl-Carreau.py`): pseudo-time stepping with adaptive `dt` based on Newton iteration count
- **Evolution scripts** (`aorta-2d-cyl-NS-evol.py`, `aorta-2d-cyl-Carreau-evol.py`): fixed time stepping with `dt = 0.01`
- **Scheme**: Implicit Euler (first-order, unconditionally stable)

### Nonlinear Solver
- **Method**: Newton with inexact Krylov solve
- **Linear solver**: MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
- **Tolerances**: atol = 1e-13, rtol = 1e-13
- **Max Newton iterations**: 10–16

### Pulsatile Inlet Profile (Evolution Codes)
Inlet velocity varies with period T = 1.0:
$$v_{\text{in}}(t) = \begin{cases}
-\frac{4V}{t_{\rm max}^2} t(t - t_{\rm max}) & \text{if } t \bmod T < t_{\rm max} \\
0 & \text{otherwise}
\end{cases}$$
with parameters V = 0.7, $t_{\rm max}$ = 0.3 (adjustable in code).

## Dependencies

- **FEniCS** (2019.1.0 or compatible)
- **PETSc** with MUMPS support
- **Python 3** with NumPy, SciPy
- **MPI** for parallel execution (optional)

## License

Academic research code. Use for educational and research purposes.
