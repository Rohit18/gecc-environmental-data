"""Pure functions for the final step (fractions from class-area tables). Used by finalize.py and tests."""
import pandas as pd


def fractions(urban, rural, veg_classes, fill_classes=(0, 250)):
    """urban: DataFrame [piece_id, year, glc_class, area_m2]; rural: [ua, year, glc_class, area_m2].
    Fill classes are removed from numerator and denominator. Returns (urban_frac df, rural_frac df)."""
    if not veg_classes:
        raise ValueError("vegetation class list is empty")
    out = []
    for df, key in ((urban, "piece_id"), (rural, "ua")):
        d = df[~df.glc_class.isin(fill_classes)]
        tot = d.groupby([key, "year"]).area_m2.sum().rename("denom_m2")
        veg = d[d.glc_class.isin(veg_classes)].groupby([key, "year"]).area_m2.sum().rename("veg_m2")
        f = pd.concat([tot, veg], axis=1).fillna({"veg_m2": 0.0})
        f["frac"] = f.veg_m2 / f.denom_m2
        out.append(f.reset_index())
    return out[0], out[1]
