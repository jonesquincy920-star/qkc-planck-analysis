"""Download and load Planck CMB FITS maps."""

import os
import requests
import healpy as hp
import numpy as np
from tqdm import tqdm

PLANCK_BASE_URL = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/fits/"

PLANCK_MAPS = {
    "SMICA": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    "NILC":  "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
    "SEVEM": "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def download_planck_map(method: str = "SMICA", data_dir: str = DATA_DIR) -> str:
    """Download a Planck 2018 CMB map if not already cached. Returns local path."""
    os.makedirs(data_dir, exist_ok=True)
    filename = PLANCK_MAPS[method.upper()]
    local_path = os.path.join(data_dir, filename)

    if os.path.exists(local_path):
        return local_path

    url = PLANCK_BASE_URL + filename
    print(f"Downloading {method} map from Planck archive...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(local_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return local_path


def load_map(path: str, field: int = 0, nside_out: int = None) -> np.ndarray:
    """Load a HEALPix map from FITS. field=0 is temperature (I). Optionally downgrade nside."""
    m = hp.read_map(path, field=field, dtype=np.float64)
    if nside_out is not None:
        current_nside = hp.get_nside(m)
        if nside_out != current_nside:
            m = hp.ud_grade(m, nside_out)
    return m
