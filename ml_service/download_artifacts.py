"""Download model weights, grid artifacts, and NetCDF if URLs are set.

Local files are left in place when they already exist.
Used on Render boot because GitHub does not store *.keras / *.npy / *.nc.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent
FORCE = os.environ.get("FORCE_DOWNLOAD", "").strip() in {"1", "true", "yes"}

TARGETS = [
    ("MODEL_URL", ML_DIR / "models" / "seasonal_cnn.keras"),
    ("SPATIAL_MASK_URL", ML_DIR / "artifacts" / "spatial_mask.npy"),
    ("LATITUDE_GRID_URL", ML_DIR / "artifacts" / "latitude_grid.npy"),
    ("LONGITUDE_GRID_URL", ML_DIR / "artifacts" / "longitude_grid.npy"),
    (
        "NC_URL",
        REPO_ROOT
        / "data"
        / "processed"
        / "sea_ice_aoi"
        / "sea_ice_bharati_2023_2025.nc",
    ),
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  saved {dest.stat().st_size} bytes", flush=True)


def main() -> int:
    missing_local = []
    for env_key, dest in TARGETS:
        url = (os.environ.get(env_key) or "").strip()
        if dest.exists() and not FORCE:
            print(f"OK  {dest.name} already present", flush=True)
            continue
        if url:
            try:
                download(url, dest)
            except Exception as exc:
                print(f"FAIL {env_key}: {exc}", file=sys.stderr, flush=True)
                return 1
            continue
        missing_local.append(str(dest))

    if missing_local:
        print(
            "Warning: these files are missing and no download URL was set:\n  "
            + "\n  ".join(missing_local)
            + "\n/predict will return 503 until they exist.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
