"""Tunggu kuota FIRMS pulih, lalu ulangi jendela jalur utara yang belum ada (laju pelan)."""
import sys, os, time, requests, yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\File_Kuliah_Semester7\Proyek-Sains-Data"
sys.path.insert(0, ROOT); os.chdir(ROOT)
from src.config import load_config
from src.download_firms import _date_windows, _fetch_window
from src.utils import ensure_dir

cfg = load_config(); key = cfg["firms"]["map_key"]
bbox = "104.6,-2.66,106.2,-2.38"
out_dir = ensure_dir("data/raw/firms_oki_north")

def quota():
    try:
        j = requests.get(f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={key}", timeout=60).json()
        return j["current_transactions"]
    except Exception:
        return 99999

for rnd in range(4):
    while quota() > 1500:
        time.sleep(30)
    wins = [w for w in _date_windows(date(2015, 1, 1), date(2024, 12, 31), 5)
            if not (out_dir / f"firms_{w[0]}_{w[1]}.csv").exists()]
    print(f"putaran {rnd+1}: {len(wins)} jendela tersisa (kuota terpakai {quota()})", flush=True)
    if not wins:
        break
    bad = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for fu in as_completed([ex.submit(_fetch_window, cfg, key, bbox, a, b, out_dir) for a, b in wins]):
            if fu.result().startswith(("error", "invalid")):
                bad += 1
    print(f"putaran {rnd+1}: gagal {bad}", flush=True)
    if bad == 0:
        break
left = [w for w in _date_windows(date(2015, 1, 1), date(2024, 12, 31), 5) if not (out_dir / f"firms_{w[0]}_{w[1]}.csv").exists()]
print("SISA GAGAL:", len(left), [str(a) for a, b in left][:20], flush=True)
