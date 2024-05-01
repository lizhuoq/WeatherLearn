import cdsapi
import os

# upper
c = cdsapi.Client()
target_dir = "target_data"

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'geopotential', 'specific_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '50', '100', '150',
            '200', '250', '300',
            '400', '500', '600',
            '700', '850', '925',
            '1000',
        ],
        'year': '2018',
        'month': '09',
        'day': '27',
        'time': '13:00',
    },
    os.path.join(target_dir, 'target_upper.nc'))


# surface
c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'mean_sea_level_pressure',
        ],
        'year': '2018',
        'month': '09',
        'day': '27',
        'time': '13:00',
    },
    os.path.join(target_dir, 'target_surface.nc'))