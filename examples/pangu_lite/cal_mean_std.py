import xarray as xr
import numpy as np
from tqdm import tqdm

from os.path import join
from os import listdir
import pickle


def cal_mean(filenames, surface_variables, pLevel=None):
    surface_map = {v: {"batch": 0, "sumV": 0} for v in surface_variables}
    for file in tqdm(filenames):
        ds = xr.open_dataset(file)
        # upper_air
        if pLevel is not None:
            ds = ds.sel(level=pLevel)

        for v in surface_variables:
            data = ds[v].data
            batch = (data != np.nan).sum()
            sumV = np.nansum(data)

            surface_map[v]["batch"] += batch
            surface_map[v]["sumV"] += sumV

    surface_mean = {v: surface_map[v]["sumV"] / surface_map[v]["batch"] for v in surface_map}
    if pLevel is None:
        return surface_mean
    
    # upper_air
    return {pLevel: surface_mean}


def cal_std(filenames, surface_variables, surface_mean, pLevel=None):
    surface_map = {v: {"batch": 0, "sumV": 0} for v in surface_variables}
    for file in tqdm(filenames):
        ds = xr.open_dataset(file)
        # upper_air
        if pLevel is not None:
            ds = ds.sel(level=pLevel)

        for v in surface_variables:
            data = ds[v].data
            batch = (data != np.nan).sum()
            sumV = np.nansum(np.abs(data - surface_mean[v]) ** 2)

            surface_map[v]["batch"] += batch
            surface_map[v]["sumV"] += sumV

    surface_std = {v: np.sqrt(surface_map[v]["sumV"] / surface_map[v]["batch"]) for v in surface_map}
    if pLevel is None:
        return surface_std
    
    # upper_air
    return {pLevel: surface_std}


if __name__ == "__main__":
    dataset_root = "data"
    # cal surface mean std
    filenames = []
    for status in ["train", "valid", "test"]:
        for file in listdir(join(dataset_root, status)):
            if file.startswith("surface"):
                filenames.append(join(dataset_root, status, file))

    surface_variables = ["u10", "v10", "t2m", "msl"]

    ## cal surface mean
    surface_mean = cal_mean(filenames, surface_variables)
    
    ## cal surface std
    surface_std = cal_std(filenames, surface_variables, surface_mean)

    ## save surface mean
    with open(join(dataset_root, "surface_mean.pkl"), "wb") as f:
        pickle.dump(surface_mean, f)

    ## save surface std
    with open(join(dataset_root, "surface_std.pkl"), "wb") as f:
        pickle.dump(surface_std, f)

    # cal upper_air 
    filenames = []
    for status in ["train", "valid", "test"]:
        for file in listdir(join(dataset_root, status)):
            if file.startswith("upper_air"):
                filenames.append(join(dataset_root, status, file))

    pLevels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    upper_air_variables = ['z', 'q', 't', 'u', 'v']

    ## cal upper_air mean
    upper_air_mean = {}
    for pl in pLevels:
        upper_air_mean = {**upper_air_mean, **cal_mean(filenames, upper_air_variables, pl)}

    ## cal upper_air std
    upper_air_std = {}
    for pl in pLevels:
        upper_air_std = {**upper_air_std, **cal_std(filenames, upper_air_variables, upper_air_mean[pl], pl)}

    ## save upper_air mean
    with open(join(dataset_root, "upper_air_mean.pkl"), "wb") as f:
        pickle.dump(upper_air_mean, f)

    ## save upper_air std
    with open(join(dataset_root, "upper_air_std.pkl"), "wb") as f:
        pickle.dump(upper_air_std, f)
