"""Inspect boundary layers: feature count, CRS, ID fields, invalid geometries, bounds.

Writes manifests/boundary_layers.json and manifests/boundary_files.csv (path, bytes, sha256),
plus manifests/ua_uac20_vs_corrected.csv (row-level differences between the two UA files).
Read-only on inputs. Logs to logs/inspect_boundaries.log.
"""
import glob, hashlib, json, logging, os, time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = "/path/to/scratch/urbveg"
B = f"{ROOT}/data/raw/boundaries"
logging.basicConfig(filename=f"{ROOT}/logs/inspect_boundaries.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def summarize(name, gdf, path):
    t = time.time()
    valid = shapely.is_valid(gdf.geometry.values)
    reasons = pd.Series(shapely.is_valid_reason(gdf.geometry.values[~valid])).str.split("[").str[0]
    d = {
        "layer": name, "path": path, "n_features": int(len(gdf)),
        "crs": gdf.crs.to_string() if gdf.crs else None,
        "crs_name": gdf.crs.name if gdf.crs else None,
        "columns": [c for c in gdf.columns if c != "geometry"],
        "geom_types": gdf.geom_type.value_counts().to_dict(),
        "n_invalid": int((~valid).sum()),
        "invalid_reasons": reasons.value_counts().to_dict(),
        "n_empty": int(gdf.geometry.is_empty.sum()),
        "n_null_geom": int(gdf.geometry.isna().sum()),
        "total_bounds": [float(x) for x in gdf.total_bounds],
    }
    log.info("%s summarized in %.1fs", name, time.time() - t)
    return d


out, files = {}, []

# --- tracts (52 TIGER/Line state files) ---
shps = sorted(glob.glob(f"{B}/tracts_2020/tl_2020_*_tract/tl_2020_*_tract.shp"))
tr = pd.concat([gpd.read_file(p) for p in shps], ignore_index=True)
d = summarize("tracts_2020", tr, f"{B}/tracts_2020/tl_2020_*_tract/*.shp ({len(shps)} files)")
d["n_files"] = len(shps)
d["n_states_fips"] = int(tr.STATEFP.nunique())
d["statefp"] = sorted(tr.STATEFP.unique().tolist())
d["geoid_unique"] = bool(tr.GEOID.is_unique)
d["geoid_len"] = tr.GEOID.str.len().value_counts().to_dict()
d["n_aland_zero"] = int((tr.ALAND == 0).sum())
d["n_per_state"] = tr.STATEFP.value_counts().sort_index().to_dict()
out["tracts_2020"] = d

# --- Urban Areas: base file and corrected file ---
ua = {}
for k in ["tl_2020_us_uac20", "tl_2020_us_uac20_corrected"]:
    p = f"{B}/urban_areas_2020/{k}/{k}.shp"
    g = gpd.read_file(p)
    ua[k] = g
    d = summarize(k, g, p)
    idc = [c for c in g.columns if c.upper().startswith("GEOID") or c.upper().startswith("UACE")]
    d["id_fields"] = idc
    for c in idc:
        d[f"{c}_unique"] = bool(g[c].is_unique)
    if "NAME20" in g.columns:
        pr = g[g.NAME20.str.contains(r", PR\b|PR--|--PR", regex=True)]
        d["n_ua_name_contains_PR"] = int(len(pr))
        d["pr_examples"] = pr.NAME20.head(8).tolist()
    for c in ["UATYP20", "LSAD20", "FUNCSTAT20", "MTFCC20"]:
        if c in g.columns:
            d[f"{c}_counts"] = g[c].value_counts().to_dict()
    out[k] = d

# row-level comparison base vs corrected
a, b = ua["tl_2020_us_uac20"], ua["tl_2020_us_uac20_corrected"]
key = "GEOID20" if "GEOID20" in a.columns else a.columns[0]
m = a[[key, "NAME20", "ALAND20", "geometry"]].merge(
    b[[key, "NAME20", "ALAND20", "geometry"]], on=key, how="outer", suffixes=("_base", "_corr"), indicator=True)
both = m["_merge"] == "both"
ga = gpd.GeoSeries(m.loc[both, "geometry_base"], crs=a.crs)
gb = gpd.GeoSeries(m.loc[both, "geometry_corr"], crs=a.crs)
m["geom_equal"] = pd.NA
m.loc[both, "geom_equal"] = shapely.equals(ga.values, gb.values)
m["aland_equal"] = m["ALAND20_base"] == m["ALAND20_corr"]
m["name_equal"] = m["NAME20_base"] == m["NAME20_corr"]
diff = m[(m["_merge"] != "both") | (m["geom_equal"] == False) | (~m["aland_equal"]) | (~m["name_equal"])]
diff.drop(columns=["geometry_base", "geometry_corr"]).to_csv(f"{ROOT}/manifests/ua_uac20_vs_corrected.csv", index=False)
out["ua_comparison"] = {"key": key, "n_base": len(a), "n_corrected": len(b),
                        "n_only_base": int((m._merge == "left_only").sum()),
                        "n_only_corrected": int((m._merge == "right_only").sum()),
                        "n_rows_differing": int(len(diff)),
                        "differing_names": diff["NAME20_base"].fillna(diff["NAME20_corr"]).head(40).tolist()}

# --- June 2023 regional update files ---
for k in ["2020_UA_ATLANTA_GAINESVILLE_JUNE_2023_UPDATE", "2020_UA_NEW_ORLEANS_JUNE_2023_UPDATE"]:
    p = f"{B}/urban_areas_2020/{k}/{k}.shp"
    g = gpd.read_file(p)
    d = summarize(k, g, p)
    d["records"] = g.drop(columns="geometry").astype(str).to_dict("records")
    # compare to corrected national file
    kk = [c for c in g.columns if c.upper().startswith("GEOID")]
    if kk:
        gc = b.set_index(key)
        d["compare_to_corrected"] = []
        for _, r in g.iterrows():
            gid = r[kk[0]]
            if gid in gc.index:
                d["compare_to_corrected"].append({"geoid": gid, "geom_equal_corrected": bool(
                    shapely.equals(r.geometry, gc.loc[gid, "geometry"])),
                    "geom_equal_base": bool(shapely.equals(r.geometry, a.set_index(key).loc[gid, "geometry"]))
                    if gid in set(a[key]) else None})
            else:
                d["compare_to_corrected"].append({"geoid": gid, "in_corrected": False})
    out[k] = d

# --- existing FUA file (search result; not copied) ---
fua_p = "/path/to/home/GECC/FUA_polygons/FUA_polygons.shp"
if os.path.exists(fua_p):
    try:
        g = gpd.read_file(fua_p); enc = "as declared (.cpg UTF-8)"
    except UnicodeDecodeError:
        # .cpg says UTF-8 but the .dbf has non-UTF-8 bytes; latin1 fallback for inspection only
        g = gpd.read_file(fua_p, encoding="latin1"); enc = "latin1 fallback (.cpg UTF-8 is wrong)"
    d = summarize("fua_existing", g, fua_p)
    d["encoding_used"] = enc
    d["head"] = g.drop(columns="geometry").head(3).astype(str).to_dict("records")
    out["fua_existing"] = d

json.dump(out, open(f"{ROOT}/manifests/boundary_layers.json", "w"), indent=1, default=str)

# --- file manifest with checksums (boundary files are small) ---
rows = []
for p in sorted(glob.glob(f"{B}/**/*", recursive=True)):
    if os.path.isfile(p):
        rows.append({"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)})
pd.DataFrame(rows).to_csv(f"{ROOT}/manifests/boundary_files.csv", index=False)
log.info("done; %d files in manifest", len(rows))
print(json.dumps({k: {kk: v[kk] for kk in ("n_features", "crs", "n_invalid", "total_bounds") if kk in v}
                  for k, v in out.items()}, indent=1))
