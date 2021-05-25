# chargepol

Python code that infer charge layer polarity from lightning flashes.

Minimum requirements: numpy v.1.16.4, glob2 v.0.7, h5py v.2.9.0, sklearn v.0.21.2. Code has not been tested with other package versions.

It uses VHF Lightning Mapping Array Level 2 HDF5 files obtained from lmatools as input. Please refer to lmatools to convert LMA Level 1 data to Level 2 (i.e., process LMA source data into flash datasets).

Change parameters in the beggining of the code (lines 29-62) accordingly. 

Usage: python chargepol.py

Output: NetCDF file with polarity of a charge layer, time of a charge layer in seconds after 0 UTC, charge layer bottom altitude in km, charge layer width in km, east-west distance from network center in km, south-north distance from network center in km.

Reference: Medina et al., submitted to AGU ESS
