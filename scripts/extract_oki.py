import sys
import geopandas as gpd

src, dst = sys.argv[1], sys.argv[2]
g = gpd.read_file(src)
print("kolom:", list(g.columns), "| jumlah:", len(g))
cand = g[g["shapeName"].str.contains("Komering", case=False, na=False)]
print("kandidat:", cand["shapeName"].tolist())
oki = g[g["shapeName"].str.strip().str.lower() == "ogan komering ilir"]
assert len(oki) == 1, "OKI tidak unik / tidak ditemukan"
oki = oki.to_crs(4326)
area = oki.to_crs(6933).area.iloc[0] / 1e6  # equal-area (km2)
print("OKI bounds (lon/lat):", [round(v, 3) for v in oki.total_bounds])
print("OKI luas: %.0f km2" % area)
oki[["shapeName", "geometry"]].to_file(dst, driver="GeoJSON")
print("tersimpan:", dst)
