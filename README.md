# rtmod
Scripts for running the Sellers and SPARTACUS Radiative Transfer models using input data from MODIS, ALS and/or ICESat-2

Uses the Sellers Python implementation built by Tristan Quaife (https://github.com/tquaife/pySellersTwoStream) and SPARTACUS Fortran built by Robin Hogan (https://github.com/ecmwf/spartacus-surface).

## Key scripts:
### alsVatl08.py
Map ALS and ICESat-2 ATL08 canopy height and cover over Finland.

### config.nam
Configuration file for SPARTACUS initialisation

### final_plot.py
Evaluate RT model performance with respect to MODIS land surface albedo.

### getICESat2metrics.py
Script for extracting canopy height and cover from ICESat-2 ATL08 data.

### globalLAImap.py
Warp GEDI L3 canopy height onto the MODIS grid.

### laiMap.py
Plot ALS and ICESat-2 canopy height and cover over Finland where MODIS >= 1 and <= 3.

### latitudeGradients.py
Assess latitudinal graidents in MODIS LAI and ALS canopy height.

### rtmod.py
Run Sellers and SPARTACUS over MODIS pixels driven by MODIS, ALS and/or ICESat-2 data.
