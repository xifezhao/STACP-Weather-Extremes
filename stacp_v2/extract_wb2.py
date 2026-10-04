"""Extract ERA5 / Pangu-Weather / IFS-ENS fields from WeatherBench2 (public GCS).

Output (/content/stacp_data):
  era5_{var}.npz   : 00 UTC daily fields 1990-01-01 .. 2022-12-31, lat 25..70N, all lon
  pangu_{var}.npz  : 00 UTC initialisations 2018-2022, leads 72/120/168 h
  ens_{var}.npz    : IFS-ENS 50-member 90th percentile and maximum, 2022 inits every
                     2nd day, leads 72/120/168 h
Units: T2m in K; Z500 converted from geopotential (m2 s-2) to geopotential height (m)
by dividing by g = 9.80665 for ALL sources.
"""
import os, time, sys
import numpy as np
import xarray as xr
import dask

dask.config.set(scheduler="threads", num_workers=32)
OUT = "/content/stacp_data"
os.makedirs(OUT, exist_ok=True)
G = 9.80665
SO = {"token": "anon"}
P = {
    "era5": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
    "pangu": "gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr",
    "ens": "gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr",
}
LEADS_H = [72, 120, 168]
LAT = slice(25, 70)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sel_var(ds, var):
    if var == "t2m":
        return ds["2m_temperature"], 1.0
    return ds["geopotential"].sel(level=500), 1.0 / G


def era5(var):
    fn = f"{OUT}/era5_{var}.npz"
    if os.path.exists(fn):
        return
    ds = xr.open_zarr(P["era5"], storage_options=SO)
    da, fac = sel_var(ds, var)
    chunks, times = [], []
    for yr in range(1990, 2023):
        x = da.sel(time=slice(f"{yr}-01-01T00", f"{yr}-12-31T00", 4)).sel(latitude=LAT)
        x = x.transpose("time", "latitude", "longitude")
        chunks.append((x.values * fac).astype(np.float32))
        times.append(x.time.values)
        log("era5", var, yr, chunks[-1].shape)
    np.savez(fn, data=np.concatenate(chunks), time=np.concatenate(times),
             lat=x.latitude.values, lon=x.longitude.values)


def pangu(var):
    fn = f"{OUT}/pangu_{var}.npz"
    if os.path.exists(fn):
        return
    ds = xr.open_zarr(P["pangu"], storage_options=SO, decode_timedelta=True)
    da, fac = sel_var(ds, var)
    td = (ds.prediction_timedelta.values / np.timedelta64(1, "h")).astype(int)
    li = [int(np.where(td == h)[0][0]) for h in LEADS_H]
    out, times = [], []
    for yr in range(2018, 2023):
        x = da.isel(prediction_timedelta=li).sel(time=slice(f"{yr}-01-01T00", f"{yr}-12-31T00", 2)).sel(latitude=LAT)
        x = x.transpose("time", "prediction_timedelta", "latitude", "longitude")
        out.append((x.values * fac).astype(np.float32))
        times.append(x.time.values)
        log("pangu", var, yr, out[-1].shape)
    np.savez(fn, data=np.concatenate(out), time=np.concatenate(times), leads=np.array(LEADS_H))


def ens(var):
    fn = f"{OUT}/ens_{var}.npz"
    if os.path.exists(fn):
        return
    ds = xr.open_zarr(P["ens"], storage_options=SO, decode_timedelta=True)
    da, fac = sel_var(ds, var)
    td = (ds.prediction_timedelta.values / np.timedelta64(1, "h")).astype(int)
    li = [int(np.where(td == h)[0][0]) for h in LEADS_H]
    q90, mx, times = [], [], []
    for mo in range(1, 13):
        end = f"2022-{mo + 1:02d}-01T00" if mo < 12 else "2023-01-01T00"
        x = da.isel(prediction_timedelta=li).sel(time=slice(f"2022-{mo:02d}-01T00", end)).sel(latitude=LAT)
        t = x.time.values
        keep = [i for i, tt in enumerate(t) if tt.astype("datetime64[h]").astype(int) % 24 == 0
                and ((tt.astype("datetime64[D]") - np.datetime64("2022-01-01")).astype(int) % 2 == 0)
                and tt < np.datetime64(end)]
        x = x.isel(time=keep).transpose("time", "number", "prediction_timedelta", "latitude", "longitude")
        v = x.values * fac
        q90.append(np.quantile(v, 0.9, axis=1).astype(np.float32))
        mx.append(v.max(axis=1).astype(np.float32))
        times.append(x.time.values)
        log("ens", var, mo, v.shape)
    np.savez(fn, q90=np.concatenate(q90), max=np.concatenate(mx), time=np.concatenate(times),
             leads=np.array(LEADS_H))


if __name__ == "__main__":
    which = sys.argv[1:] or ["era5", "pangu", "ens"]
    for var in ["t2m", "z500"]:
        for w in which:
            t0 = time.time()
            {"era5": era5, "pangu": pangu, "ens": ens}[w](var)
            log("DONE", w, var, f"{time.time() - t0:.0f}s")
    log("ALL DONE")
