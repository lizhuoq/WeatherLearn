from os import listdir
from os.path import join
import pickle
from datetime import datetime
from typing import Literal

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose
import torch
import xarray as xr
import numpy as np
from dateutil.relativedelta import relativedelta


def surface_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        surface_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        surface_std = pickle.load(f)

    mean_seq, std_seq, channel_seq = [], [], []
    variables = sorted(list(surface_mean.keys()))
    for v in variables:
        channel_seq.append(v)
        mean_seq.append(surface_mean[v])
        std_seq.append(surface_std[v])

    return Normalize(mean_seq, std_seq), channel_seq


def upper_air_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        upper_air_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        upper_air_std = pickle.load(f)

    pLevels = sorted(list(upper_air_mean.keys()))
    variables = sorted(list(list(upper_air_mean.values())[0].keys()))
    normalize = {}
    for pl in pLevels:
        mean_seq, std_seq = [], []
        for v in variables:
            mean_seq.append(upper_air_mean[pl][v])
            std_seq.append(upper_air_std[pl][v])

        normalize[pl] = Normalize(mean_seq, std_seq)

    return normalize, variables, pLevels


def surface_inv_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        surface_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        surface_std = pickle.load(f)

    mean_seq, std_seq, channel_seq = [], [], []
    variables = sorted(list(surface_mean.keys()))
    for v in variables:
        channel_seq.append(v)
        mean_seq.append(surface_mean[v])
        std_seq.append(surface_std[v])

    invTrans = Compose([
        Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]), 
        Normalize([-x for x in mean_seq], [1.] * len(std_seq))
    ])
    return invTrans, channel_seq


def upper_air_inv_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        upper_air_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        upper_air_std = pickle.load(f)

    pLevels = sorted(list(upper_air_mean.keys()))
    variables = sorted(list(list(upper_air_mean.values())[0].keys()))
    normalize = {}
    for pl in pLevels:
        mean_seq, std_seq = [], []
        for v in variables:
            mean_seq.append(upper_air_mean[pl][v])
            std_seq.append(upper_air_std[pl][v])

        invTrans = Compose([
            Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]), 
            Normalize([-x for x in mean_seq], [1.] * len(std_seq))
        ])
        normalize[pl] = invTrans

    return normalize, variables, pLevels


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, flag: Literal["train", "test", "valid"]):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.flag = flag
        self.surface_filenames = [join(dataset_dir, flag, x) for x in listdir(join(dataset_dir, flag)) if x.startswith("surface")]
        self.upper_air_filenames = [join(dataset_dir, flag, f"upper_air_{y}_{str(m).zfill(2)}.nc") 
                                    for y, m in [x.split(".")[-2].split("_")[-2:] for x in self.surface_filenames]]
        self.surface_transform, self.surface_variables = surface_transform(join(dataset_dir, "surface_mean.pkl"), 
                                                                           join(dataset_dir, "surface_std.pkl"))
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(dataset_dir, "upper_air_mean.pkl"), 
                                                                                                         join(dataset_dir, "upper_air_std.pkl"))
        
        times = [datetime.strptime(x.split(".")[-2][-7:], "%Y_%m") for x in self.surface_filenames]
        st = min(times)  # include
        et = max(times) + relativedelta(months=+1)  # exclude

        self.date = np.arange(f"{st.year}-{str(st.month).zfill(2)}", f"{et.year}-{str(et.month).zfill(2)}", dtype='datetime64[D]')

        self.land_mask, self.soil_type, self.topography = self._load_constant_mask()

    def __getitem__(self, index):
        surface_t, upper_air_t = self._get_data(index)
        surface_t_1, upper_air_t_1 = self._get_data(index + 1)
        if self.flag == "train":
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, torch.tensor([
            self.date[index].astype(int), self.date[index + 1].astype(int)
        ])

    def _get_data(self, index):
        date = self.date[index]
        year = date.astype("datetime64[Y]").astype(int) + 1970
        month = date.astype("datetime64[M]").astype(int) % 12 + 1
        day = (date - date.astype("datetime64[M]")).astype(int) + 1

        surface_file = join(self.dataset_dir, self.flag, f"surface_{year}_{str(month).zfill(2)}.nc")
        upper_air_file = join(self.dataset_dir, self.flag, f"upper_air_{year}_{str(month).zfill(2)}.nc")

        surface_ds = xr.open_dataset(surface_file).sel(time=f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}")
        upper_air_ds = xr.open_dataset(upper_air_file).sel(time=f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}")

        surface_data = np.stack([surface_ds[x].data for x in self.surface_variables], axis=0)  # C Lat Lon
        surface_data = torch.from_numpy(surface_data.astype(np.float32))
        surface_data = self.surface_transform(surface_data)

        upper_air_data = torch.stack([self.upper_air_transform[pl](
            torch.from_numpy(np.stack([upper_air_ds.sel(level=pl)[x].data for x in self.upper_air_variables], axis=0).astype(np.float32))
        ) for pl in self.upper_air_pLevels], dim=1)  # C Pl Lat Lon
        return surface_data, upper_air_data

    def __len__(self):
        return len(self.date) - 1

    def _load_constant_mask(self):
        mask_dir = "constant_mask"
        land_mask = torch.from_numpy(np.load(join(self.dataset_dir, mask_dir, "land_mask.npy")).astype(np.float32))
        soil_type = torch.from_numpy(np.load(join(self.dataset_dir, mask_dir, "soil_type.npy")).astype(np.float32))
        topography = torch.from_numpy(np.load(join(self.dataset_dir, mask_dir, "topography.npy")).astype(np.float32))

        return land_mask, soil_type, topography
    
    def get_constant_mask(self):
        return self.land_mask, self.soil_type, self.topography

    def get_lat_lon(self):
        example = self.surface_filenames[0]
        ds = xr.open_dataset(example)
        return ds["latitude"].data, ds["longitude"].data


def get_year_month_day(dt):
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    month = dt.astype("datetime64[M]").astype(int) % 12 + 1
    day = (dt.astype("datetime64[D]") - dt.astype("datetime64[M]")).astype(int) + 1
    return year, month, day
