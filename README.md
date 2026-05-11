#  Effects of fluid rheology and wall slip on blood flow

This repository contain codes and necessary files to reproduce data published in the paper
**_Effects of fluid rheology and wall slip on blood flow characteristics in arteries_**

Computational meshes for 3D problems can be found at TODO: add doi

## Prerequisities

The code is written in Python3 and requires the following:

1. **Legacy FEniCS 2019.1.0**  
   Installation guide can be found at [FEniCS Project Archive](https://fenicsproject.org/download/archive/).

2. **`ns_aneurysm` Package** (for 3D aneurysm simulations) 
   The package can be found on our [GitLab repository](https://gitlab.karlin.mff.cuni.cz/bio/aneurysm). To install it, use the following command:
   ```bash
   python3 -m pip install -e .
    ```

## Aortic root axisymmetric simulations

Simulations of blood flow in an idealized 2D axisymmetric aortic root geometry.
The code solves the Navier–Stokes or Carreau constitutive equations for steady-state (long-time) and pulsatile (time-dependent) flows.
Results can be reproduced by running the scripts in the folder `aortic_root`.

**Available codes:**
- `aorta-2d-cyl-NS.py`: Newtonian steady-state solver
- `aorta-2d-cyl-Carreau.py`: Carreau non-Newtonian steady-state solver
- `aorta-2d-cyl-NS-evol.py`: Newtonian time-evolution (pulsatile) solver
- `aorta-2d-cyl-Carreau-evol.py`: Carreau time-evolution (pulsatile) solver

**Mesh geometries** (parameterized by sinus radius):
- `mesh_R12.h5`: Radius 12 mm
- `mesh_R16.h5`: Radius 16 mm
- `mesh_R20.h5`: Radius 20 mm

**Parameters to set (command-line arguments):**
- `kappa`: Navier-slip coefficient (e.g., 2 Pa·s/m)
- `output_file`: Output filename for scalar results
- `beta`: Nitsche penalty parameter (e.g., 1000)
- `radius`: Sinus radius in mm (12, 16, or 20) — selects mesh file
- `hematocrit`: Blood hematocrit level (optional; 25, 45, or 65 %; default 45)

**Example: Steady-state Newtonian simulation (radius 16 mm, hematocrit 45%)**
```bash
python3 aorta-2d-cyl-NS.py 2 output_NS_45.txt 1000 16 45
```

**Example: Time-evolved Carreau simulation (radius 16 mm, hematocrit 45%)**
```bash
python3 aorta-2d-cyl-Carreau-evol.py 2 output_Carreau_evol.txt 1000 16 45
```

Time-evolution scripts output velocity and pressure in VTK format to folders `results_NS_<hematocrit>/` and `results_Carreau_<hematocrit>/` (viewable with ParaView).

For detailed description of parameters, rheological models, and numerical methods, see `aortic_root/README.md`.



## Patient-specific aneurysm simulations

Simulations of blood flow in a patient-specific 3D aneurysm geometry.

The flow solver is based on the `ns_aneurysm` package and computes velocity and pressure for Newtonian and Carreau blood rheology models with either no-slip or partial-slip wall boundary conditions. Post-processing scripts compute wall shear stress, surface-based hemodynamic indicators, and scalar flow metrics.

Results can be reproduced by running the scripts in the folder `aneurysm`.

**Available codes:**

- `aneurysm_example.py`: 3D patient-specific aneurysm flow solver
- `compute_wss_aneurysm_generalized.py`: vectorial wall shear stress computation
- `evaluate_indicators.py`: evaluation of surface-based hemodynamic indicators
- `compute_flow_metrics.py`: spatial integration of hemodynamic quantities over the aneurysm region

**Main parameters:**

- `model`: blood rheology model, for example `Newtonian`, `Carreau_HCT25`, `Carreau_HCT45`, or `Carreau_HCT65`
- `theta`: slip parameter; use `-1.0` for Dirichlet no-slip and values in `[0, 1]` for Nitsche slip enforcement
- `meshname`: name of the aneurysm mesh file used by the flow solver
- `mesh`: path to the marked aneurysm mesh in HDF5 format
- `w_file`: path to the HDF5 file containing velocity and pressure
- `res_folder`: folder where simulation and post-processing results are written
- `case`: aneurysm case identifier; use always `case01`

**Example: Partial-slip workflow**

```bash
model=Carreau_HCT45
theta=0.8
meshname=case01_uniform_200um
meshfolder=meshes/
mesh=meshes/case01_uniform_200um_marked.h5
dest=results/
w_file=${dest}/${meshname}/pulsatile/FacetNormal_th_none/Nitsche_Navier_slip_80/${model}/w.h5
res_folder=${dest}/wss/

python3 aneurysm_example.py \
    -model ${model} \
    -mu 0.00345 \
    -rho 1050 \
    -Theta ${theta} \
    -meshname ${meshname} \
    -meshfolder meshes/ \
    -element th \
    -normal FacetNormal \
    -stab none \
    -refsys_filename meshes/case01_refsystems_SI.dat \
    -profile pulsatile \
    -profile_analytical False \
    -inflow_file meshes/inletcase01.dat \
    -periods 3 \
    -uniform_dt_last_period \
    -uniform_dt 0.01 \
    -unit_system SI \
    -bcout_dir_do_nothing \
    -dest ${res_folder}

python3 compute_wss_aneurysm_generalized.py \
    --mesh ${mesh} \
    --element th \
    --stab none \
    --w-file ${w_file} \
    -o ${res_folder} \
    --theta ${theta} \
    --model ${model}

python3 evaluate_indicators.py \
    --case ${case} \
    --mesh-folder ${mesh} \
    --element th \
    --edgelengths 200 \
    --res-folder ${res_folder} \
    --rescaled

python3 compute_flow_metrics.py \
    --case ${case} \
    --mesh-folder ${mesh} \
    --element th \
    --res-folder ${res_folder} \
    --v-file v_cp.xdmf \
    --wss-file wss_rescaled_v_tan_cp.xdmf \
    --theta ${theta} \
    --model ${model} \
    --wss-degree 2
```

Time-dependent velocity, pressure, wall shear stress, and surface indicators are stored in XDMF format and can be visualized in ParaView. Scalar flow metrics are written to output tables.

For a detailed description of the aneurysm workflow, input files, post-processing steps, and no-slip setup, see `aneurysm/README.md`.