"""Gambar peta grid 5 km di atas batas Kab. OKI.

Menandai sel yang titik tengahnya di dalam OKI, sel di luar OKI, dan bagian
OKI yang belum tercakup kotak grid. Hasil: presentasi/aset_m34/peta_grid_5km.png
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.grid import build_grid, grid_spec  # noqa: E402

cfg = load_config()
g = build_grid(cfg)
spec = grid_spec(cfg)
oki = gpd.read_file(ROOT / "data/external/admin/oki_boundary.geojson").to_crs(4326)
P = oki.geometry.iloc[0]
g["inside"] = [P.contains(Point(x, y)) for x, y in zip(g.lon_center, g.lat_center)]

top = spec["lat_min"] + spec["n_rows"] * spec["step"]
right = spec["lon_min"] + spec["n_cols"] * spec["step"]
north = gpd.GeoSeries([P.intersection(box(spec["lon_min"] - 1, top, right + 1, top + 2))], crs=4326)
pct = north.to_crs(6933).area.iloc[0] / oki.to_crs(6933).area.iloc[0] * 100
n_in = int(g.inside.sum())
print(f"{len(g)} sel, {n_in} di dalam OKI, ujung utara belum tercakup {pct:.1f}%")

plt.rcParams["font.family"] = "Calibri"
fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=250)
for r in g.itertuples():
    ax.add_patch(Rectangle((r.lon_min, r.lat_min), r.lon_max - r.lon_min, r.lat_max - r.lat_min,
                           facecolor="#95C4A6" if r.inside else "#EEF1EF",
                           edgecolor="white", linewidth=0.35))
north.plot(ax=ax, facecolor="#FBE3D6", edgecolor="#D9480F", hatch="////", linewidth=0)
oki.boundary.plot(ax=ax, color="#1B4332", linewidth=1.4)
ax.set_xlim(spec["lon_min"] - 0.02, right + 0.02)
ax.set_ylim(spec["lat_min"] - 0.02, -2.33)
ax.set_aspect("equal")
ax.set_xticks([104.5, 105, 105.5, 106, 106.5])
ax.set_xticklabels(["104,5", "105,0", "105,5", "106,0", "106,5"])
ax.set_yticks([-4.0, -3.5, -3.0, -2.5])
ax.set_yticklabels(["4,0", "3,5", "3,0", "2,5"])
ax.tick_params(labelsize=9, colors="#6C757D", length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Bujur (°BT)", fontsize=9, color="#6C757D")
ax.set_ylabel("Lintang (°LS)", fontsize=9, color="#6C757D")
handles = [
    Patch(facecolor="#95C4A6", edgecolor="none", label=f"Sel di dalam OKI ({n_in})"),
    Patch(facecolor="#EEF1EF", edgecolor="#CED4DA", label=f"Sel di luar OKI ({len(g) - n_in})"),
    Patch(facecolor="#FBE3D6", edgecolor="#D9480F", hatch="////",
          label=f"Ujung utara belum tercakup ({pct:.1f}%)".replace(".", ",")),
    Line2D([0], [0], color="#1B4332", lw=1.4, label="Batas Kab. OKI (BPS 2020)"),
]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2,
          frameon=False, fontsize=9, handlelength=1.3, columnspacing=1.5)
fig.tight_layout()
out = ROOT / "presentasi/aset_m34/peta_grid_5km.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=250, facecolor="white")
print("tersimpan:", out)
