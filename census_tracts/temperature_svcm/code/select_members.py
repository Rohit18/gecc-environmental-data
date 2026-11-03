"""List each yearly archive and select the daily members for JJA and same-year DJF.

Member names look like <archive-dir>/..._<YYYY><TMAX|TMIN>_<DDD>.tif, where DDD is the
calendar day-of-year. Availability is taken from the archive listing, never assumed."""
import datetime as dt, json, os, re, subprocess, sys
from pathlib import Path

PAT = re.compile(r"_(\d{4})(TMAX|TMIN)_(\d{3})\.tif$")

def season_dates(year, season):
    if season == "JJA":
        d0, d1 = dt.date(year, 6, 1), dt.date(year, 8, 31)
        return [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1)]
    feb_end = dt.date(year, 3, 1) - dt.timedelta(1)
    jan_feb = [dt.date(year, 1, 1) + dt.timedelta(n) for n in range((feb_end - dt.date(year, 1, 1)).days + 1)]
    dec = [dt.date(year, 12, d) for d in range(1, 32)]
    return jan_feb + dec

def main(year, source_dir, out_dir):
    archive = Path(source_dir) / f"NorthAmerica{year}.tar"
    listing = subprocess.run(["tar", "-tvf", str(archive)], capture_output=True, text=True, check=True).stdout
    members = {}
    for line in listing.splitlines():
        parts = line.split()
        name, size = parts[-1], int(parts[2])
        m = PAT.search(name)
        if m and int(m.group(1)) == year:
            doy = int(m.group(3))
            date = dt.date(year, 1, 1) + dt.timedelta(doy - 1)
            members[(m.group(2), date)] = {"member": name, "bytes": size}
    sel = {"year": year, "archive": archive.name, "n_members_in_archive": len(members), "seasons": {}}
    want = []
    for season in ("DJF", "JJA"):
        dates = season_dates(year, season)
        for measure in ("TMAX", "TMIN"):
            have = [d for d in dates if (measure, d) in members]
            miss = [d.isoformat() for d in dates if (measure, d) not in members]
            sel["seasons"][f"{season}_{measure}"] = {
                "expected_calendar_days": len(dates), "available_source_days": len(have),
                "missing_source_dates": miss,
                "members": [{"date": d.isoformat(), **members[(measure, d)]} for d in have]}
            want += [members[(measure, d)]["member"] for d in have]
    Path(out_dir, f"selection_{year}.json").write_text(json.dumps(sel, indent=1))
    Path(out_dir, f"members_{year}.txt").write_text("\n".join(sorted(set(want))) + "\n")
    print(year, {k: (v["available_source_days"], v["expected_calendar_days"]) for k, v in sel["seasons"].items()})

if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
