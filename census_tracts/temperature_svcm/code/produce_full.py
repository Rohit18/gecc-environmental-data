"""Build the 2003-2024 seasonal tract table from NERSC composites and tract geography."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import geopandas as gpd
W = Path(sys.argv[1]); TRACTS = sys.argv[2]; OUT = Path(sys.argv[3]); YEARS = range(2003, 2025)
sys.path.insert(0, str(W / "code")); from reducers import reduce
COLUMNS = ["GEOID","StFIPS","CoFIPS","TrFIPS","LAT_GeoCen","LON_GeoCen","Area_Land","Area_Water","Pop",
           "observation_year","season","tmax_mean_c","tmax_stddev_c","tmax_median_c","tmax_valid_count",
           "tmin_mean_c","tmin_stddev_c","tmin_median_c","tmin_valid_count"]

def comp_path(year, season, meas):
    hits = sorted(W.glob(f"work/composites*/composite_{year}_{season}_{meas}.npy"))
    if not hits:
        raise FileNotFoundError(f"composite {year} {season} {meas}")
    return hits[0]

def fmt(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))

def main():
    if (OUT / "CensusTract_air_temperature_2003_2024.csv").exists():
        raise FileExistsError("Refusing to overwrite an existing temperature table")
    geo = pd.read_csv(W / "work/grid_index_geoids.csv", dtype=str).GEOID.values; n = len(geo)
    ix = np.load(W / "work/sampling_index_final.npz")
    t, ni, cov = ix["tract"], ix["native_idx"], ix["coverage"]
    if not np.isfinite(cov).all() or np.any(cov < np.float32(1.0 / 255.0)) or np.any(cov > 1):
        raise ValueError("Production index must have coverage >= 1/255 and <= 1")
    attrs = gpd.read_file(TRACTS, columns=COLUMNS[:9])
    attrs["GEOID"] = attrs["GEOID"].astype(str).str.zfill(11)
    for col, start, end in (("StFIPS", 0, 2), ("CoFIPS", 2, 5), ("TrFIPS", 5, 11)):
        attrs[col] = attrs["GEOID"].str.slice(start, end)
    if attrs.GEOID.nunique() != n or set(attrs.GEOID) != set(geo):
        raise ValueError("Tract geometry does not match the sampling index")
    prefix = {
        row.GEOID: ",".join(str(getattr(row, col)) for col in COLUMNS[:9])
        for row in attrs.itertuples(index=False)
    }
    res = {}
    for year in YEARS:
        for season in ("DJF", "JJA"):
            for meas in ("TMAX", "TMIN"):
                res[(year, season, meas)] = reduce(np.load(comp_path(year, season, meas)), t, ni, n, cov)
        print("reduced", year, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "CensusTract_air_temperature_2003_2024.csv"
    order = np.argsort(geo, kind="stable")
    with open(out, "wb") as f:
        f.write((",".join(COLUMNS) + "\r\n").encode())
        for i in order:
            for year in YEARS:
                for season in ("DJF", "JJA"):
                    vals = []
                    for meas in ("TMAX", "TMIN"):
                        o = res[(year, season, meas)]; c = int(o["count"][i])
                        vals += ["", "", "", "0"] if c == 0 else [fmt(o["mean_weighted"][i]), fmt(o["sd_pop"][i]), fmt(o["median"][i]), str(c)]
                    f.write((f"{prefix[geo[i]]},{year},{season}," + ",".join(vals) + "\r\n").encode())
    print("wrote", out.name)

if __name__ == "__main__":
    main()
