# vscode Relative path
import sys
sys.path.append("../")

from os.path import join
from os import makedirs
import argparse

import onnxruntime as ort
from tqdm import tqdm
import xarray as xr
import numpy as np
import torch

from examples.pangu_lite.data_utils import DatasetFromFolder, surface_inv_transform, upper_air_inv_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead_time", type=int, help="Must be a multiple of 24, such as 24, 48, n * 24.", required=True)
    opt = parser.parse_args()

    LEAD_TIME = opt.lead_time
    assert LEAD_TIME % 24 == 0, "Must be a multiple of 24, such as 24, 48, n * 24."
    num_iterations = LEAD_TIME // 24

    pangu_lite_root = "../examples/pangu_lite"
    test_set = DatasetFromFolder(join(pangu_lite_root, "data"), "test")
    lat, lon = test_set.get_lat_lon()
    surface_invTrans, surface_variables = surface_inv_transform(join(pangu_lite_root, "data/surface_mean.pkl"),  
                                                                join(pangu_lite_root, "data/surface_std.pkl"))
    upper_air_invTrans, upper_air_variables, pLevels = upper_air_inv_transform(join(pangu_lite_root, "data/upper_air_mean.pkl"),  
                                                                            join(pangu_lite_root, "data/upper_air_std.pkl"))

    surface_ordered_variables = ["msl", "u10", "v10", "t2m"]
    upper_air_ordered_variables = ["z", "q", "t", "u", "v"]
    ordered_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

    output_list = []

    split = 0
    save_root = f"bench/pangu_{LEAD_TIME}"
    makedirs(save_root, exist_ok=True)

    for input_surface, input_upper_air, _, _, times in tqdm(test_set):
        input_surface = surface_invTrans(input_surface)  # C Lat Lon
        input_upper_air = torch.stack([upper_air_invTrans[pl](input_upper_air[:, i, :, :]) for i, pl in enumerate(pLevels)], dim=1)  # C Pl Lat Lon

        # ordered
        input_surface = torch.stack([input_surface[surface_variables.index(v), :, :] 
                                    for v in surface_ordered_variables], dim=0)
        ## upper air channels
        input_upper_air = torch.stack([input_upper_air[upper_air_variables.index(v), :, :, :]
                                    for v in upper_air_ordered_variables], dim=0)
        ## upper air pLevels
        input_upper_air = torch.stack([input_upper_air[:, pLevels.index(l), :, :] 
                                    for l in ordered_levels], dim=1)
        
        # numpy
        input_surface = input_surface.numpy().astype(np.float32)
        input_upper_air = input_upper_air.numpy().astype(np.float32)

        for _ in range(num_iterations):
            output_upper_air, output_surface = ort_session_24.run(None, {'input':input_upper_air, 'input_surface':input_surface})

            input_upper_air = output_upper_air
            input_surface = output_surface

        init_time = times[0].item()

        ds = xr.Dataset(
            data_vars=dict(
                **{v: (["time", "prediction_timedelta", "latitude", "longitude"], output_surface[i, :, :][np.newaxis, np.newaxis, :, :]) 
                   for i, v in enumerate(surface_ordered_variables)}, 
                **{v: (["time", "prediction_timedelta", "level", "latitude", "longitude"], output_upper_air[i, :, :, :][np.newaxis, np.newaxis, :, :, :]) 
                   for i, v in enumerate(upper_air_ordered_variables)}
            ), 
            coords={
                "level": ordered_levels, 
                "latitude": lat, 
                "longitude": lon, 
                "prediction_timedelta": np.array([num_iterations]).astype("timedelta64[D]"), 
                "time": np.array([init_time]).astype("datetime64[D]")
            }
        )

        output_list.append(ds)

        if len(output_list) >= 30:
            pangu_ds = xr.concat(output_list, dim="time")
            pangu_ds.to_zarr(join(save_root, f"{split}.zarr"))

            split += 1
            output_list = []

    pangu_ds = xr.concat(output_list, dim="time")
    pangu_ds.to_zarr(join(save_root, f"{split}.zarr"))
