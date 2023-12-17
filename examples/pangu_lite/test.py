from typing import Literal
import argparse
import os
import sys
sys.path.append("../../")

from weatherlearn.models import Pangu_lite
from data_utils import DatasetFromFolder, surface_inv_transform, upper_air_inv_transform

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import xarray as xr


def inference(input_surface: torch.Tensor, input_upper_air: torch.Tensor, surface_mask: torch.Tensor, 
              model_name: str, device: Literal["GPU", "CPU"], lead_time: int):
    """Inference
    Args:
        input_surface: input surface tensor, need to be normalized to N(0, 1), shape: B C Lat Lon.
        input_upper_air: input upper air tensor, need to be normalized to N(0, 1), shape: B C Pl Lat Lon.
        surface_mask: surface mask tensor.
        model_name: the storage location of the model file.
        device: GPU or CPU.
        lead_time: Must be a multiple of 24, such as 24, 48, n * 24.
    """
    pangu_lite24 = Pangu_lite()
    
    pangu_lite24.eval()

    is_gpu = True if device == "GPU" else False

    if is_gpu:
        pangu_lite24.cuda()
        pangu_lite24.load_state_dict(torch.load(model_name))
    else:
        pangu_lite24.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))

    surface_invTrans, surface_variables = surface_inv_transform("data/surface_mean.pkl", "data/surface_std.pkl")
    upper_air_invTrans, upper_air_variables, pLevels = upper_air_inv_transform("data/upper_air_mean.pkl", "data/upper_air_std.pkl")

    num_iterations = lead_time // 24

    for _ in range(num_iterations):
        output_surface, output_upper_air = pangu_lite24(input_surface, surface_mask, input_upper_air)

        input_surface = output_surface
        input_upper_air = output_upper_air

    output_surface = surface_invTrans(output_surface).squeeze(0)  # C Lat Lon
    output_upper_air = torch.stack([upper_air_invTrans[pl](output_upper_air[:, :, i, :, :]) for i, pl in enumerate(pLevels)], dim=2)  # C Pl Lat Lon

    return output_surface.detach().cpu().numpy(), output_upper_air.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="GPU", type=str, choices=["GPU", "CPU"])
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--lead_time", type=int, help="Must be a multiple of 24, such as 24, 48, n * 24.")
    opt = parser.parse_args()

    test_set = DatasetFromFolder("data", "test")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    land_mask, soil_type, topography = test_set.get_constant_mask()
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0)

    lat, lon = test_set.get_lat_lon()

    surface_invTrans, surface_variables = surface_inv_transform("data/surface_mean.pkl", "data/surface_std.pkl")
    upper_air_invTrans, upper_air_variables, pLevels = upper_air_inv_transform("data/upper_air_mean.pkl", "data/upper_air_std.pkl")

    DEVICE = opt.device
    MODEL_NAME = opt.model_name
    LEAD_TIME = opt.lead_time
    is_gpu = True if DEVICE == "GPU" else False
    assert LEAD_TIME % 24 == 0, "Must be a multiple of 24, such as 24, 48, n * 24."

    pangu_lite = Pangu_lite()
    pangu_lite.eval()
    if is_gpu:
        pangu_lite.cuda()
        pangu_lite.load_state_dict(torch.load(MODEL_NAME))
        surface_mask = surface_mask.cuda()
    else:
        pangu_lite.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

    num_iterations = LEAD_TIME // 24

    test_bar = tqdm(test_loader)
    output_list = []

    split = 0
    save_root = "bench"
    os.makedirs(save_root, exist_ok=True)

    for input_surface, input_upper_air, target_surface, target_upper_air, times in test_bar:
        if is_gpu:
            input_surface = input_surface.cuda()
            input_upper_air = input_upper_air.cuda()

        for _ in range(num_iterations):
            output_surface, output_upper_air = pangu_lite(input_surface, surface_mask, input_upper_air)
            input_surface = output_surface
            input_upper_air = output_upper_air

        init_time = times.squeeze(0)[0].item()
        output_surface = surface_invTrans(output_surface).squeeze(0)  # C Lat Lon
        output_upper_air = torch.stack([upper_air_invTrans[pl](output_upper_air[:, :, i, :, :]) for i, pl in enumerate(pLevels)], dim=2).squeeze(0)  # C Pl Lat Lon

        output_surface = output_surface.detach().cpu().numpy()
        output_upper_air = output_upper_air.detach().cpu().numpy()

        ds = xr.Dataset(
            data_vars=dict(
                **{v: (["time", "prediction_timedelta", "latitude", "longitude"], output_surface[i, :, :][np.newaxis, np.newaxis, :, :]) 
                   for i, v in enumerate(surface_variables)}, 
                **{v: (["time", "prediction_timedelta", "level", "latitude", "longitude"], output_upper_air[i, :, :, :][np.newaxis, np.newaxis, :, :, :]) 
                   for i, v in enumerate(upper_air_variables)}
            ), 
            coords={
                "level": pLevels, 
                "latitude": lat, 
                "longitude": lon, 
                "prediction_timedelta": np.array([num_iterations]).astype("timedelta64[D]"), 
                "time": np.array([init_time]).astype("datetime64[D]")
            }
        )

        output_list.append(ds)

        if len(output_list) >= 30:
            forecast_ds = xr.concat(output_list, dim="time")
            forecast_ds.to_zarr(os.path.join(save_root, f"forecast_{LEAD_TIME}_{split}.zarr"))

            split += 1
            output_list = []

    forecast_ds = xr.concat(output_list, dim="time")
    forecast_ds.to_zarr(os.path.join(save_root, f"forecast_{LEAD_TIME}_{split}.zarr"))
