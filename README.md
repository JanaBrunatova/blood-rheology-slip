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

Results for the idealized aortic root can be reproduced by running the following commands in the folder `aortic_root`:
```bash
python3
```

## Patient-specific aneurysm simulations

Navier-Stokes equations are solved for velocity and pressure by running the code `aneurysm_example.py` in the folder `aneurysm`

**Parameters to set:**

- `model`: Rheological model for blood. Either `Newtonian`, `Carreau_HCT25`, `Carreau_HCT45`, or `Carreau_HCT65`.
- `Theta`: Slip parameter between 0 an 1. Use `-1.0` for Dirichlet no-slip BC.
- `meshname`: Name of the `.xml` mesh file.
- `meshfolder`: Folder containing the mesh file.
- `dest`: Name of the folder where the results will be stored.

```bash
python3 aneurysm_example.py -model ${model} -mu 0.00345 -rho 1050 -Theta ${Theta} -meshname ${mesh} -meshfolder ../meshes/ -element th -normal FacetNormal -stab none -refsys_filename ../meshes/case01_refsystems_SI.dat -profile pulsatile -profile_analytical False -inflow_file ../meshes/inletcase01.dat -periods 3 -uniform_dt_last_period -uniform_dt 0.01 -unit_system SI -bcout_dir_do_nothing -dest ${dest}
```
The postprocessing is split into 3 steps. In the first step, vectorial WSS is evaluated. 
```bash
python3 compute_wss_aneurysm.py --mesh-folder ${meshfolder} --res-folder ${res_folder} --element th --stab none
```
Next, the hemodynamic quantities of interest (living on the surface mesh) are evaluated and stored.
```bash
python3 evaluate_indicators.py
```
And finally, the hemodynamic quantities are spatially integrated to obtain corresponding scalar indices
```bash
python3 compute_flow_metrics.py
```
