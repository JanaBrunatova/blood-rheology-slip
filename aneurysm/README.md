# Patient-specific aneurysm simulations

This folder contains scripts for patient-specific 3D aneurysm simulations and post-processing. The workflow solves the Navier–Stokes equations for velocity and pressure, computes wall shear stress, evaluates surface hemodynamic indicators, and integrates selected quantities over the aneurysm region.

The simulations support Newtonian and Carreau blood rheology models with either no-slip or partial-slip wall boundary conditions.

## Available codes

- `aneurysm_example.py`: 3D patient-specific aneurysm flow solver
- `compute_wss_aneurysm_generalized.py`: wall shear stress computation
- `evaluate_indicators.py`: evaluation of surface hemodynamic indicators
- `compute_flow_metrics.py`: integration of hemodynamic quantities into scalar indices

## Input files

The workflow requires:

- patient-specific aneurysm mesh
- marked mesh file in HDF5 format
- inlet flow profile
- local reference-system file
- velocity-pressure solution file `w.h5`
- selected rheological model
- selected wall-slip parameter

The mesh and flow-profile files are stored in the `meshes/` folder.

## Rheological models

The blood rheology model is selected with the `model` parameter.

Available options are:

- `Newtonian`
- `Carreau_HCT25`
- `Carreau_HCT45`
- `Carreau_HCT65`

The Carreau models correspond to hematocrit levels of 25 %, 45 %, and 65 %.

## Wall boundary conditions

The wall boundary condition is controlled by the slip parameter `Theta` in the flow simulation and by `theta` during post-processing.

Use:

- `Theta = -1.0` for Dirichlet no-slip boundary conditions
- `0 <= Theta <= 1` for partial-slip boundary conditions enforced by Nitsche's method

Use the same slip value during all post-processing steps as in the original flow simulation.

## Step 1: Run the flow simulation

Velocity and pressure are computed by running `aneurysm_example.py`.

Set the following variables first:

```bash
model=Newtonian
Theta=-1.0
meshname=case01_uniform_200um.xml
meshfolder=meshes/
dest=results/Newtonian_Dirichlet_noslip
```

Run:

```bash
python3 aneurysm_example.py \
    -model ${model} \
    -mu 0.00345 \
    -rho 1050 \
    -Theta ${Theta} \
    -meshname ${meshname} \
    -meshfolder ${meshfolder} \
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
    -dest ${dest}
```

The simulation stores velocity and pressure in the destination folder. The file `w.h5` is used in the post-processing steps below.

### Parameters

| Parameter | Description | Example |
|---|---|---|
| `model` | Blood rheology model | `Newtonian` |
| `mu` | Dynamic viscosity used for the Newtonian model | `0.00345` |
| `rho` | Blood density | `1050` |
| `Theta` | Slip parameter | `-1.0` |
| `meshname` | Name of the aneurysm mesh file | `case01_uniform_200um.xml` |
| `meshfolder` | Folder containing the mesh file | `meshes/` |
| `refsys_filename` | File containing local reference systems | `meshes/case01_refsystems_SI.dat` |
| `inflow_file` | File containing the prescribed inlet flow profile | `meshes/inletcase01.dat` |
| `periods` | Number of cardiac cycles to simulate | `3` |
| `uniform_dt` | Time-step size used during the last cardiac cycle | `0.01` |
| `dest` | Folder where velocity and pressure are stored | `results/Newtonian_Dirichlet_noslip` |

## Step 2: Compute wall shear stress

Wall shear stress is computed from the velocity-pressure solution stored in `w.h5`.

Set the following variables first:

```bash
mesh=meshes/case01_uniform_200um_marked.h5
w_file=results/Newtonian_Dirichlet_noslip/w.h5
res_folder=results/Newtonian_Dirichlet_noslip/
theta=-1.0
model=Newtonian
```

Run:

```bash
python3 compute_wss_aneurysm_generalized.py \
    --mesh ${mesh} \
    --element th \
    --stab none \
    --w-file ${w_file} \
    -o ${res_folder} \
    --theta ${theta} \
    --model ${model}
```

### Parameters

| Argument | Description | Example |
|---|---|---|
| `--mesh` | Path to the marked aneurysm mesh in HDF5 format | `meshes/case01_uniform_200um_marked.h5` |
| `--element` | Finite element used in the flow solve | `th` |
| `--stab` | Stabilization method | `none` |
| `--w-file` | HDF5 file containing velocity and pressure | `results/Newtonian_Dirichlet_noslip/w.h5` |
| `-o`, `--res-folder` | Folder where WSS results are written | `results/Newtonian_Dirichlet_noslip/` |
| `--theta` | Slip parameter used in the simulation | `-1.0` |
| `--model` | Blood rheology model | `Newtonian` |

This step writes wall shear stress fields to `${res_folder}`. For partial-slip simulations, it also computes WSS based on the rescaled tangential velocity.

## Step 3: Evaluate hemodynamic indicators

The script `evaluate_indicators.py` computes hemodynamic quantities living on the aneurysm surface.

For no-slip simulations, run:

```bash
python3 evaluate_indicators.py \
    --mesh-folder ${mesh} \
    --element th \
    --edgelengths 200 \
    --res-folder ${res_folder}
```

For partial-slip simulations, add the `--rescaled` argument:

```bash
python3 evaluate_indicators.py \
    --mesh-folder ${mesh} \
    --element th \
    --edgelengths 200 \
    --res-folder ${res_folder} \
    --rescaled
```

### Parameters

| Argument | Description | Example |
|---|---|---|
| `--mesh-folder` | Path to the marked aneurysm mesh | `meshes/case01_uniform_200um_marked.h5` |
| `--element` | Finite element used in the flow solve | `th` |
| `--edgelengths` | Edge-length value used for indicator evaluation | `200` |
| `--res-folder` | Folder containing WSS files and receiving indicator output | `results/Newtonian_Dirichlet_noslip/` |
| `--rescaled` | Uses rescaled tangential-velocity WSS for partial-slip simulations | optional flag |

Use `--rescaled` only for partial-slip simulations.

## Step 4: Compute scalar flow metrics

The script `compute_flow_metrics.py` spatially integrates the computed hemodynamic quantities and writes scalar indices.

For no-slip simulations, run:

```bash
python3 compute_flow_metrics.py \
    --mesh-folder ${mesh} \
    --element th \
    --res-folder ${res_folder} \
    --mu 0.00345 \
    --v-file v_cp.xdmf \
    --wss-file wss_standard_CG_1_cp.xdmf \
    --theta ${theta} \
    --wss-degree 1
```

For partial-slip simulations, run:

```bash
python3 compute_flow_metrics.py \
    --mesh-folder ${mesh} \
    --element th \
    --res-folder ${res_folder} \
    --v-file v_cp.xdmf \
    --wss-file wss_rescaled_v_tan_cp.xdmf \
    --theta ${theta} \
    --wss-degree 2
```

### Parameters

| Argument | Description | Example |
|---|---|---|
| `--mesh-folder` | Path to the marked aneurysm mesh | `meshes/case01_uniform_200um_marked.h5` |
| `--element` | Finite element used in the flow solve | `th` |
| `--res-folder` | Folder containing velocity, pressure, WSS, and indicator files | `results/Newtonian_Dirichlet_noslip/` |
| `--mu` | Dynamic viscosity used for Newtonian simulations | `0.00345` |
| `--v-file` | Velocity file | `v_cp.xdmf` |
| `--wss-file` | WSS file used for scalar metric computation | `wss_standard_CG_1_cp.xdmf` |
| `--theta` | Slip parameter used in the simulation | `-1.0` |
| `--wss-degree` | Polynomial degree of the WSS field | `1` |

For no-slip simulations, use `wss_standard_CG_1_cp.xdmf`.

For partial-slip simulations, use `wss_rescaled_v_tan_cp.xdmf`.

## Complete no-slip example

```bash
model=Newtonian
Theta=-1.0
theta=-1.0
meshname=case01_uniform_200um.xml
meshfolder=meshes/
mesh=meshes/case01_uniform_200um_marked.h5
dest=results/Newtonian_Dirichlet_noslip
w_file=${dest}/w.h5
res_folder=${dest}/

python3 aneurysm_example.py \
    -model ${model} \
    -mu 0.00345 \
    -rho 1050 \
    -Theta ${Theta} \
    -meshname ${meshname} \
    -meshfolder ${meshfolder} \
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
    -dest ${dest}

python3 compute_wss_aneurysm_generalized.py \
    --mesh ${mesh} \
    --element th \
    --stab none \
    --w-file ${w_file} \
    -o ${res_folder} \
    --theta ${theta} \
    --model ${model}

python3 evaluate_indicators.py \
    --mesh-folder ${mesh} \
    --element th \
    --edgelengths 200 \
    --res-folder ${res_folder}

python3 compute_flow_metrics.py \
    --mesh-folder ${mesh} \
    --element th \
    --res-folder ${res_folder} \
    --mu 0.00345 \
    --v-file v_cp.xdmf \
    --wss-file wss_standard_CG_1_cp.xdmf \
    --theta ${theta} \
    --wss-degree 1
```

## Complete partial-slip example

```bash
model=Carreau_HCT45
Theta=0.5
theta=0.5
meshname=case01_uniform_200um.xml
meshfolder=meshes/
mesh=meshes/case01_uniform_200um_marked.h5
dest=results/Carreau_HCT45_partial_slip
w_file=${dest}/w.h5
res_folder=${dest}/

python3 aneurysm_example.py \
    -model ${model} \
    -mu 0.00345 \
    -rho 1050 \
    -Theta ${Theta} \
    -meshname ${meshname} \
    -meshfolder ${meshfolder} \
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
    -dest ${dest}

python3 compute_wss_aneurysm_generalized.py \
    --mesh ${mesh} \
    --element th \
    --stab none \
    --w-file ${w_file} \
    -o ${res_folder} \
    --theta ${theta} \
    --model ${model}

python3 evaluate_indicators.py \
    --mesh-folder ${mesh} \
    --element th \
    --edgelengths 200 \
    --res-folder ${res_folder} \
    --rescaled

python3 compute_flow_metrics.py \
    --mesh-folder ${mesh} \
    --element th \
    --res-folder ${res_folder} \
    --v-file v_cp.xdmf \
    --wss-file wss_rescaled_v_tan_cp.xdmf \
    --theta ${theta} \
    --wss-degree 2
```

## Output files

The workflow produces:

- velocity and pressure files from the flow simulation
- wall shear stress fields in XDMF format
- surface hemodynamic indicator fields in XDMF format
- scalar flow metrics stored in output tables

The XDMF files can be visualized in ParaView.