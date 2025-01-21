# PAM

This module takes pellet information (size, velocity, material, layers), plasma information ne,Te, geometry,
and computes a predicted pellet ablation and density source.

* start with (geqdsk, ne, Te) or ods

* input desired pellet information

* compute pellet ablation and density source

## Running PAM

Place the files needed in the current folder (see the `pam_validation` folder).

To execute PAM run in that folder:
```
python3.7 ../pam/run_pam.py
```
