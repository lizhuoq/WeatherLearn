## Data
`input_surface.npy` stores the input surface variables. It is a numpy array shaped (4,721,1440) where the first dimension represents the 4 surface variables (MSLP, U10, V10, T2M in the exact order).

`input_upper.npy` stores the upper-air variables. It is a numpy array shaped (5,13,721,1440) where the first dimension represents the 5 surface variables (Z, Q, T, U and V in the exact order), and the second dimension represents the 13 pressure levels (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa in the exact order).

`input_surface.npy` and `input_upper.npy`, which correspond to the ERA5 initial fields of at 12:00UTC, 2018/09/27.
`target_surface.npy` and `target_upper.npy`, which correspond to the ERA5 initial fields of at 13:00UTC, 2018/09/27.

`input_surface.npy`: [Google drive](https://drive.google.com/file/d/1pj8QEVNpC1FyJfUabDpV4oU3NpSe0BkD/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1i4o5i8guAqmOus6PWncAlA?pwd=4z9s)

`input_upper.npy`: [Google drive](https://drive.google.com/file/d/1--7xEBJt79E3oixizr8oFmK_haDE77SS/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1mS8X5MqEdbVfF2u2Us62FQ?pwd=sgx6)

`target_surface.npy`: [Baidu netdisk](https://pan.baidu.com/s/1949LEf-KnEn4ocCPQ8ulvA?pwd=8d8x)

`target_upper.npy`: [Baidu netdisk](https://pan.baidu.com/s/1moI2wcQl9vdOEsMMZi2tLA?pwd=8mr7)

## Downloading trained models
The 1-hour model (pangu_weather_1.onnx): [Google drive](https://drive.google.com/file/d/1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1M7SAigVsCSH8hpw6DE8TDQ?pwd=ie0h)


## Optimization details
- learning rate: 1e-6
- finetune epochs: 100
- lradj: cosine

### Results
`slurm-12359622.out` and `train_results.json`
