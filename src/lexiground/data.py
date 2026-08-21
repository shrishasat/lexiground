
from pathlib import Path
from urllib.request import urlretrieve

from platformdirs import user_cache_dir


LANCASTER_URL = "https://osf.io/48wsc/download"

ICONICITY_URL = (
    "https://raw.githubusercontent.com/"
    "bodowinter/iconicity_ratings/"
    "master/ratings/iconicity_ratings_cleaned.csv"
)


def get_cache_dir():

    cache_dir = Path(
        user_cache_dir("lexiground")
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
    
# ============================================================
# iconicity
# ============================================================
    
def get_iconicity_path():

    cache_dir = get_cache_dir()

    path = (
        cache_dir
        / "iconicity_ratings_cleaned.csv"
    )

    if path.exists():

        return path

    return download_file(
        ICONICITY_URL,
        path,
    )
