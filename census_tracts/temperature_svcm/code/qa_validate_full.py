"""Validate the NERSC seasonal tract table and write input QA records."""
import datetime as dt, hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np, pandas as pd
W = Path(sys.argv[1]); OUT = Path(sys.argv[2]); TRACTS = sys.argv[3]
YEARS = list(range(2003, 2025))
COLUMNS = ["GEOID","StFIPS","CoFIPS","TrFIPS","LAT_GeoCen","LON_GeoCen","Area_Land","Area_Water","Pop",
           "observation_year","season","tmax_mean_c","tmax_stddev_c","tmax_median_c","tmax_valid_count",
           "tmin_mean_c","tmin_stddev_c","tmin_median_c","tmin_valid_count"]
NUM = COLUMNS[11:]
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
tab = OUT / "CensusTract_air_temperature_2003_2024.csv"

# ---------------- coverage
t = pd.read_csv(tab, dtype={"GEOID": str}, low_memory=False)
(t.groupby(["season", "observation_year"])
   .agg(records=("GEOID", "size"),
        non_empty_tmax_mean_c=("tmax_mean_c", lambda s: int(s.notna().sum())), empty_tmax_mean_c=("tmax_mean_c", lambda s: int(s.isna().sum())),
        non_empty_tmin_mean_c=("tmin_mean_c", lambda s: int(s.notna().sum())), empty_tmin_mean_c=("tmin_mean_c", lambda s: int(s.isna().sum())))
   .reset_index().to_csv(OUT / "CensusTract_air_temperature_coverage.csv", index=False))

# ---------------- audits -> manifest, daily QA, outliers
audit = []
for p in sorted(W.glob("work/composites*/daily_audit.json")):
    audit += json.loads(p.read_text())
aud = pd.DataFrame(audit)
aud["observation_year"] = aud["date"].str[:4].astype(int)
aud = aud[aud.observation_year.isin(YEARS)].drop_duplicates(subset=["member"])
aud["archive"] = "NorthAmerica" + aud["observation_year"].astype(str) + ".tar"
pd.DataFrame({"logical_path": aud["archive"] + "/" + aud["member"], "date": aud["date"], "observation_year": aud["observation_year"],
              "season": aud["season"], "measure": aud["measure"], "file_size_bytes": aud["bytes"], "sha256": aud["sha256"],
              "crs": "Sinusoidal (sphere radius 6371007.181 m)", "resolution_m": aud["res"].map(lambda r: f"{r[0]:.6f} x {abs(r[1]):.6f}"),
              "width": aud["width"], "height": aud["height"], "dtype": aud["dtype"], "nodata": aud["nodata"],
              "metadata_scale": aud["scales"].map(lambda s: s[0]), "metadata_offset": aud["offsets"].map(lambda s: s[0]),
              "applied_scale": 0.1, "applied_offset": 0.0, "used": "production"}
             ).sort_values(["observation_year", "season", "measure", "date"]).to_csv(OUT / "source_input_manifest.csv", index=False)
qa = []
for y in YEARS:
    sel = json.loads((W / f"work/lists/selection_{y}.json").read_text())
    for key, rec in sel["seasons"].items():
        season, meas = key.split("_")
        a = aud[(aud.observation_year == y) & (aud.season == season) & (aud.measure == meas)]
        notes = []
        if rec["missing_source_dates"]:
            notes.append("archive numbered day-of-year 001-365; " + ",".join(rec["missing_source_dates"]) + " absent from source; not imputed")
        nb = int(a.out_of_range_masked.sum())
        if nb: notes.append(f"{int((a.out_of_range_masked>0).sum())} daily file(s) with values outside -95..65 C masked; see outlier_masking_qa.csv")
        qa.append({"observation_year": y, "season": season, "measure": meas, "expected_calendar_days": rec["expected_calendar_days"],
                   "available_source_days": rec["available_source_days"], "missing_source_dates": ";".join(rec["missing_source_dates"]),
                   "invalid_or_out_of_range_values_masked": nb, "notes": " | ".join(notes)})
pd.DataFrame(qa).to_csv(OUT / "daily_input_qa.csv", index=False)
ix = np.load(W / "work/sampling_index_final.npz"); geo = pd.read_csv(W / "work/grid_index_geoids.csv", dtype=str).GEOID.values
oq = []
for _, a in aud[aud.out_of_range_masked > 0].sort_values(["date", "measure"]).iterrows():
    key = f"{a.observation_year}_{a.season}_{a.measure}"
    hits = sorted(W.glob(f"work/composites*/outliers_{key}_{a.date}.npy"))
    tr = []
    if hits:
        cells = np.load(hits[0]); hit = np.isin(ix["native_idx"], cells)
        tr = sorted(set(geo[np.unique(ix["tract"][hit])]))
    oq.append({"logical_path": f"{a.archive}/{a.member}", "date": a.date, "observation_year": a.observation_year, "season": a.season,
               "variable": a.measure, "values_masked": int(a.out_of_range_masked), "raw_min_masked": a.out_of_range_raw_min,
               "raw_max_masked": a.out_of_range_raw_max, "celsius_min_masked": round(a.out_of_range_raw_min * 0.1, 1),
               "celsius_max_masked": round(a.out_of_range_raw_max * 0.1, 1), "valid_pixels_before": int(a.valid_pixels_before),
               "valid_pixels_after": int(a.valid_pixels_after), "n_affected_tract_year_season": len(tr),
               "affected_tract_year_season_keys": ";".join(f"{g}|{a.observation_year}|{a.season}" for g in tr)})
pd.DataFrame(oq).to_csv(OUT / "outlier_masking_qa.csv", index=False)

# ---------------- release validation
import geopandas as gpd
V = {}
def gate(name, ok, **details):
    V[name] = {"status": "pass" if ok else "fail", **details}
tracts = gpd.read_file(TRACTS, columns=["GEOID"])
source_geoids = set(tracts.GEOID.astype(str).str.zfill(11))
gate("tract_source_unique_geoids", len(source_geoids) == 84414, observed=len(source_geoids))
gate("exact_column_names_and_order", list(t.columns) == COLUMNS)
gate("row_count", len(t) == 3714216, expected=3714216, observed=len(t))
gate("unique_geoids", t.GEOID.nunique() == 84414, observed=int(t.GEOID.nunique()))
gate("geoid_universe_matches_tract_source", set(t.GEOID) == source_geoids)
gate("geoid_11_characters", t.GEOID.str.fullmatch(r"\d{11}").all())
gate("unique_key_geoid_year_season", not t.duplicated(["GEOID", "observation_year", "season"]).any())
gate("years_and_seasons", set(t.observation_year) == set(YEARS) and set(t.season) == {"JJA", "DJF"})
invalid = 0
for measure in ("tmax", "tmin"):
    c = t[f"{measure}_valid_count"]
    stats = t[[f"{measure}_mean_c", f"{measure}_stddev_c", f"{measure}_median_c"]]
    invalid += int((c.lt(0) | c.isna() | c.mod(1).ne(0)).sum())
    invalid += int((c.eq(0) & stats.notna().any(axis=1)).sum())
    invalid += int((c.gt(0) & stats.isna().any(axis=1)).sum())
gate("count_and_missingness_rules", invalid == 0, violations=invalid)
finite = all(np.isfinite(t[col].dropna().to_numpy(dtype=float)).all() for col in NUM)
gate("numeric_values_finite", bool(finite))
temperature_columns = [f"{measure}_{stat}_c" for measure in ("tmax", "tmin") for stat in ("mean", "median")]
range_ok = all(t[col].dropna().between(-95, 65).all() for col in temperature_columns)
gate("temperature_means_and_medians_in_celsius_range", bool(range_ok))
with tab.open("rb") as source:
    crlf_lines = sum(line.endswith(b"\r\n") for line in source)
gate("line_terminator_crlf", crlf_lines == len(t) + 1)
V["_summary"] = {"gates": len(V), "failed": [k for k, val in V.items() if val["status"] == "fail"]}
(OUT / "validation.json").write_text(json.dumps(V, indent=2))
print(json.dumps(V["_summary"], indent=2))
if V["_summary"]["failed"]:
    raise SystemExit(1)
