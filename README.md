# chargepol

Python code that infers charge layer polarity and vertical location from VHF-based lightning mapping array (LMA) observations of lightning flashes.

Minimum requirements: numpy v.1.16.4, glob2 v.0.7, h5py v.2.9.0, sklearn v.0.21.2. Code has not been tested with other package versions.

It uses VHF Lightning Mapping Array Level 2 HDF5 files obtained from lmatools as input. Please refer to lmatools ( https://github.com/deeplycloudy/lmatools ) to convert LMA Level 1 data to Level 2 (i.e., process LMA source data into flash datasets).

Change parameters in the beginning of the code (lines 29-62) accordingly. 

Usage: python chargepol.py

Output: NetCDF file with polarity of a charge layer ('pos' or 'neg'), time of a charge layer in seconds after 0 UTC, charge layer bottom altitude in km, charge layer vertical depth in km, east-west distance from LMA center in km, south-north distance from LMA center in km.

Reference: 

Medina, B. L., Carey, L. D., Lang, J. T., Bitzer, P. M., Deierling, W., Zhu, Y. (2021). Characterizing charge structure in Central Argentina thunderstorms during RELAMPAGO utilizing a new charge layer polarity identification method. Earth and Space Science. In revision (pending major revisions). https://doi.org/10.1002/essoar.10506781.2
