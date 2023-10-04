# chargepol

Python code that infers charge layer polarity and vertical location from VHF-based lightning mapping array (LMA) observations of lightning flashes.

Minimum requirements: numpy v.1.25.2, xarray v.2023.8.0, sklearn v.1.3.0. Code has not been tested with other package versions.

It uses VHF Lightning Mapping Array Level 2 netcdf4 files obtained from pyxlma as input. Please refer to [xlma-python's flash_sort_grid example](https://github.com/deeplycloudy/xlma-python/blob/master/examples/pyxlma_flash_sort_grid.py) to convert LMA Level 1 data to Level 2 (i.e., process LMA source data into flash datasets).

Change parameters in the beginning of the code (lines 32-62) accordingly. 

Usage: python chargepol.py

Output: csv file with polarity of a charge layer ('pos' or 'neg'), time of a charge layer in seconds after 0 UTC, charge layer bottom altitude in km, charge layer vertical depth in km, east-west distance from LMA center in km, south-north distance from LMA center in km, longitude and latitude of flash.

Reference: 

Medina, B. L., Carey, L. D., Lang, J. T., Bitzer, P. M., Deierling, W., Zhu, Y. (2021). Characterizing charge structure in Central Argentina thunderstorms during RELAMPAGO utilizing a new charge layer polarity identification method. Earth and Space Science, v. 8, p. e2021EA001803. https://doi.org/10.1029/2021EA001803
