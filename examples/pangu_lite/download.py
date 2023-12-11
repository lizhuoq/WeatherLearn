import cdsapi

import asyncio
from os.path import join
from os import makedirs

from weatherlearn.data_utils.coroutine_download import download

YEARS = [str(x) for x in range(2007, 2020)]  # train from 2007-2017, validate on 2019, test on 2018.
MONTHS = [str(x).zfill(2) for x in range(1, 13)]
DAYS = [str(x).zfill(2) for x in range(1, 32)]
TIMES = ["00:00"]
SURFACE_VARIABLES = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure']
UPPER_AIR_VARIABLES = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
PRESSURE_LEVEL = ['50', '100', '150',
                  '200', '250', '300',
                  '400', '500', '600',
                  '700', '850', '925',
                  '1000']


async def main():
    data_root = "data"
    train_root, valid_root, test_root = [join(data_root, x) for x in ("train", "valid", "test")]

    makedirs(train_root, exist_ok=True)
    makedirs(valid_root, exist_ok=True)
    makedirs(test_root, exist_ok=True)

    train_years = YEARS[:-2]
    valid_years = YEARS[-1:]
    test_years = YEARS[-2:-1]

    for years, root in zip([train_years, valid_years, test_years], [train_root, valid_root, test_root]):
        # download surface variables
        await asyncio.gather(*[download("reanalysis-era5-single-levels", SURFACE_VARIABLES,
                                        year, month, DAYS, TIMES, join(root, f"surface_{year}_{month}.nc"))
                               for year in years for month in MONTHS])

        # download upper_air variables
        await asyncio.gather(*[download("reanalysis-era5-pressure-levels", UPPER_AIR_VARIABLES,
                                        year, month, DAYS, TIMES, join(root, f"upper_air_{year}_{month}.nc"),
                                        pressure_level=PRESSURE_LEVEL)
                               for year in years for month in MONTHS])


if __name__ == '__main__':
    asyncio.run(main())
