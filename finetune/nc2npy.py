import xarray as xr
import numpy as np
import os

target_dir = "target_data"

ds_surface = xr.open_dataset(os.path.join(target_dir, "target_surface.nc"))
# input_surface.npy stores the input surface variables. 
# It is a numpy array shaped (4,721,1440) where the first dimension represents the 4 surface variables 
# (MSLP, U10, V10, T2M in the exact order).
array_surface = np.concatenate(
    [ds_surface[v].values for v in ["msl", "u10", "v10", "t2m"]], axis=0
)

# input_upper.npy stores the upper-air variables. 
# It is a numpy array shaped (5,13,721,1440) where the first dimension represents the 5 surface variables 
# (Z, Q, T, U and V in the exact order), and the second dimension represents the 13 pressure levels 
# (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa in the exact order).
ds_upper = xr.open_dataset(os.path.join(target_dir, "target_upper.nc")).sortby("level", ascending=False)
array_upper = np.concatenate(
    [ds_upper[v].values for v in ["z", "q", "t", "u", "v"]], axis=0
)

np.save(os.path.join(target_dir, "target_surface.npy"), array_surface)
np.save(os.path.join(target_dir, "target_upper.npy"), array_upper)
