"""Unduh FIRMS untuk jalur utara OKI yang terlewat (lintang -2,66 s/d -2,38)."""
import sys, copy, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\File_Kuliah_Semester7\Proyek-Sains-Data"
sys.path.insert(0, ROOT)
import os; os.chdir(ROOT)
from src.config import load_config
from src.download_firms import _date_windows, _fetch_window
from src.utils import ensure_dir

cfg = load_config()
f = cfg["firms"]
bbox = "104.6,-2.66,106.2,-2.38"
out_dir = ensure_dir("data/raw/firms_oki_north")
start, end = date.fromisoformat(cfg["time"]["start_date"]), date.fromisoformat(cfg["time"]["end_date"])
wins = [w for w in _date_windows(start, end, 5) if not (out_dir / f"firms_{w[0]}_{w[1]}.csv").exists()]
print(f"jendela: {len(wins)} | bbox {bbox}", flush=True)
done, bad = 0, []
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(_fetch_window, cfg, f["map_key"], bbox, a, b, out_dir): (a, b) for a, b in wins}
    for fu in as_completed(futs):
        done += 1
        st = fu.result()
        if st.startswith(("error", "invalid")):
            bad.append((str(futs[fu][0]), st[:60]))
        if done % 50 == 0 or done == len(wins):
            print(f"progres {done}/{len(wins)} ({time.time()-t0:.0f} dtk)", flush=True)
print("gagal:", len(bad), bad[:10], flush=True)
print("SELESAI", flush=True)
