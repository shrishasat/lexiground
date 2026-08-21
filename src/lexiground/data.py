from pathlib import Path
from urllib.request import urlretrieve


# ============================================================
# Official data sources
# ============================================================

LANCASTER_URL = "https://osf.io/48wsc/download"

# We will add the exact official Winter URL after verifying
# the repository file path and licensing.


# ============================================================
# Cache
# ============================================================

def get_cache_dir():

    cache_dir = (
        Path.home()
        / ".cache"
        / "lexiground"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cache_dir


# ============================================================
# Download helper
# ============================================================

def download_file(
    url,
    destination,
):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading LexiGround data:\n"
        f"{url}"
    )

    urlretrieve(
        url,
        destination,
    )

    print(
        f"Saved to:\n"
        f"{destination}"
    )

    return destination


# ============================================================
# Lancaster
# ============================================================

def get_lancaster_path():

    cache_dir = get_cache_dir()

    path = (
        cache_dir
        / "lancaster_sensorimotor_norms.csv"
    )

    if path.exists():

        return path

    return download_file(
        LANCASTER_URL,
        path,
    )
