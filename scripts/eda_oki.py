"""EDA khusus wilayah administrasi Kab. OKI (batas BPS via geoBoundaries)."""
import sys, os, glob, json, math
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
from shapely.prepared import prep
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

oki = gpd.read_file("data/external/admin/oki_boundary.geojson").to_crs(4326)
OKI = oki.geometry.iloc[0]; P = prep(OKI)
S = {"oki_area_km2": round(float(oki.to_crs(6933).area.iloc[0] / 1e6))}

# ---------- FIRMS: gabung utama + jalur utara ----------
def load(folder):
    fr = []
    for f in glob.glob(folder + "/*.csv"):
        try:
            d = pd.read_csv(f)
            if not d.empty: fr.append(d)
        except Exception:
            pass
    if not fr:  # folder jalur utara tidak ada bila unduhan memakai satu kotak penuh
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time"])
    d = pd.concat(fr, ignore_index=True); d.columns = [c.strip().lower() for c in d.columns]
    return d
main = load("data/raw/firms"); north = load("data/raw/firms_oki_north")
S["files_main"] = len(glob.glob("data/raw/firms/*.csv")); S["files_north"] = len(glob.glob("data/raw/firms_oki_north/*.csv"))
key = ["latitude", "longitude", "acq_date", "acq_time"]
S["raw_dupes"] = int(main.duplicated(subset=key).sum() + north.duplicated(subset=key).sum())
raw = pd.concat([main.assign(src="main"), north.assign(src="north")], ignore_index=True).drop_duplicates(subset=key, keep="first")
pts = gpd.points_from_xy(raw["longitude"], raw["latitude"])
inside = np.array([P.contains(p) for p in pts])
S["bbox_filtered_old"] = 159399
rawo = raw[inside].copy()
S["raw_rows"] = int(len(rawo))
S["raw_rows_north_in_oki"] = int((rawo["src"] == "north").sum())
S["conf_counts"] = {str(k): int(v) for k, v in rawo["confidence"].astype(str).str.lower().value_counts().items()}
S["daynight"] = {str(k): int(v) for k, v in rawo["daynight"].value_counts().items()}
S["frp_median"] = float(rawo["frp"].median())
h = rawo[rawo["confidence"].astype(str).str.lower().isin(["n", "h", "nominal", "high"])].copy()
h["acq_date"] = pd.to_datetime(h["acq_date"])
S["filtered_rows"] = int(len(h))
ex = h[(h["acq_date"].dt.year == 2019) & (h["acq_date"].dt.month == 9)].iloc[0]
S["example_row"] = {k: str(ex[k]) for k in ["latitude", "longitude", "acq_date", "acq_time", "confidence", "bright_ti4", "frp", "daynight"]}
S["example_row"]["acq_date"] = str(ex["acq_date"].date())
h["year"] = h["acq_date"].dt.year; h["month"] = h["acq_date"].dt.month
yr = h.groupby("year").size().reindex(range(2015, 2025), fill_value=0)
S["per_year"] = {int(k): int(v) for k, v in yr.items()}
mo = h.groupby("month").size().reindex(range(1, 13), fill_value=0)
S["per_month"] = {int(k): int(v) for k, v in mo.items()}
S["share_aug_oct"] = float(h["month"].isin([8, 9, 10]).mean())
dd = h.groupby(h["acq_date"].dt.date).size()
S["days_with_fire"] = int(len(dd)); S["max_day"] = [str(dd.idxmax()), int(dd.max())]

# ---------- Grid 5 km sel di dalam OKI ----------
step = 5.0 / 111.0; lon0, lat0 = 104.4, -4.35
minx, miny, maxx, maxy = OKI.bounds
ncol = math.ceil((maxx - lon0) / step); nrow = math.ceil((maxy - lat0) / step)
cells = []
for c in range(ncol):
    for r in range(nrow):
        cx, cy = lon0 + (c + .5) * step, lat0 + (r + .5) * step
        if P.contains(gpd.points_from_xy([cx], [cy])[0]):
            cells.append((c, r, cx, cy))
cells = pd.DataFrame(cells, columns=["col", "row", "cx", "cy"])
cells["cell"] = cells["col"] * 1000 + cells["row"]
S["n_cells"] = int(len(cells))
h["col"] = np.floor((h["longitude"] - lon0) / step).astype(int); h["row"] = np.floor((h["latitude"] - lat0) / step).astype(int)
h["cell"] = h["col"] * 1000 + h["row"]
cc = h[h["cell"].isin(cells["cell"])].groupby("cell").size().sort_values(ascending=False)
S["cells_burned"] = int(len(cc))
S["share_top10pct_cells"] = float(cc.head(math.ceil(0.1 * S["n_cells"])).sum() / cc.sum())
S["max_cell_count"] = int(cc.iloc[0])

# ---------- Peta blok 2x2 (dipotong ke OKI) ----------
bc0, br0 = int((minx - lon0) // (2 * step)), int((miny - lat0) // (2 * step))
bc1, br1 = int((maxx - lon0) // (2 * step)), int((maxy - lat0) // (2 * step))
nc, nr = bc1 - bc0 + 1, br1 - br0 + 1
M = np.zeros((nr, nc), dtype=int); mask = np.zeros((nr, nc), dtype=bool)
hb = h.assign(bc=(h["col"] // 2) - bc0, br=(h["row"] // 2) - br0)
hb = hb[(hb.bc >= 0) & (hb.bc < nc) & (hb.br >= 0) & (hb.br < nr)]
for (r, c), v in hb.groupby(["br", "bc"]).size().items():
    M[r, c] = v
for r in range(nr):
    for c in range(nc):
        x0 = lon0 + (bc0 + c) * 2 * step; y0 = lat0 + (br0 + r) * 2 * step
        b = box(x0, y0, x0 + 2 * step, y0 + 2 * step)
        mask[r, c] = OKI.intersection(b).area / b.area >= 0.3
M = np.where(mask, M, 0)
nz = M[mask & (M > 0)]
S_heat = {"ncols": nc, "nrows": nr, "counts": M.tolist(), "mask": mask.astype(int).tolist(),
          "q": [int(np.quantile(nz, q)) for q in (0.25, 0.5, 0.75, 0.9)]}

# ---------- POWER titik yang dipakai sel OKI ----------
p = pd.read_csv("data/interim/power_daily.csv", parse_dates=["date"])
p["lon_s"] = (p["lon"] / 0.5).round() * 0.5; p["lat_s"] = (p["lat"] / 0.5).round() * 0.5
cells["lon_s"] = (cells["cx"] / 0.5).round() * 0.5; cells["lat_s"] = (cells["cy"] / 0.5).round() * 0.5
used = cells[["lon_s", "lat_s"]].drop_duplicates()
po = p.merge(used, on=["lon_s", "lat_s"])
S["power_points"] = int(len(used)); S["power_rows"] = int(len(po))
params = ["T2M_MAX", "T2M", "RH2M", "PRECTOTCORR", "WS2M"]
S["power_missing"] = int(po[params].isna().sum().sum())
S["power_desc"] = {c: {"min": round(float(po[c].min()), 2), "mean": round(float(po[c].mean()), 2), "max": round(float(po[c].max()), 2)} for c in params}
clim = po.groupby(po["date"].dt.month)["PRECTOTCORR"].mean()
S["clim_rain"] = {int(k): round(float(v), 2) for k, v in clim.items()}

# ---------- Panel sel OKI x hari: hari kering & hujan 30 hari ----------
dates = pd.date_range("2015-01-01", "2024-12-31")
rain = po.pivot_table(index="date", columns=["lon_s", "lat_s"], values="PRECTOTCORR").reindex(dates)
fire = h[h["cell"].isin(cells["cell"])].assign(d=h["acq_date"].dt.normalize()).groupby(["cell", "d"]).size()
rows = []
for _, cl in cells.iterrows():
    pr = rain[(cl.lon_s, cl.lat_s)].to_numpy()
    isr = (pr >= 1.0).astype(int)
    grp = np.cumsum(isr)
    dry = pd.Series((isr == 0).astype(int)).groupby(grp).cumsum().to_numpy()
    r30 = pd.Series(pr).rolling(30, min_periods=1).sum().to_numpy()
    f = np.zeros(len(dates), dtype=int)
    if cl.cell in fire.index.get_level_values(0):
        fd = fire.loc[cl.cell].index
        f[dates.get_indexer(fd)] = 1
    rows.append(pd.DataFrame({"dry": dry, "r30": r30, "fire": f}))
panel = pd.concat(rows, ignore_index=True)
S["panel_rows"] = int(len(panel)); S["prevalence"] = float(panel["fire"].mean())
b = pd.cut(panel["dry"], [-1, 0, 3, 7, 14, 30, 10000], labels=["0", "1-3", "4-7", "8-14", "15-30", ">30"])
S["fire_by_drydays"] = {str(k): round(float(v) * 100, 3) for k, v in panel.groupby(b, observed=False)["fire"].mean().items()}
b2 = pd.cut(panel["r30"], [-1, 30, 90, 180, 300, 100000], labels=["<30", "30-90", "90-180", "180-300", ">300"])
S["fire_by_rain30"] = {str(k): [round(float(v) * 100, 3), 0] for k, v in panel.groupby(b2, observed=False)["fire"].mean().items()}

json.dump({"eda": S, "heat": S_heat}, open("reports/eda_oki.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pr = {k: v for k, v in S.items() if k not in ()}
print(json.dumps(pr, indent=1, ensure_ascii=False, default=str))
print("heat", nc, "x", nr, "q", S_heat["q"], "blok OKI", int(mask.sum()))
