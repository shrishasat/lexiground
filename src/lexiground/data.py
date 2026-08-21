from pathlib import Path
from urllib.request import urlretrieve
import hashlib

from platformdirs import user_cache_dir


# ============================================================
# Official dataset sources
# ============================================================

LANCASTER_URL = (
    "https://osf.io/48wsc/download"
)

ICONICITY_URL = (
    "https://raw.githubusercontent.com/"
    "bodowinter/iconicity_ratings/"
    "master/ratings/iconicity_ratings_cleaned.csv"
)


# ============================================================
# Expected file hashes
# ============================================================

LANCASTER_SHA256 = (
    "445d363fb1f9f3e50b86d88e2f46cdc9a22b5dd8a713ce4e7be2a773d57f43c5"
)

ICONICITY_SHA256 = (
    "d66bef6a070c845dc11acaa2dfab95cae1f5db83efd687ec18f2f23a31fe6150"
)


# ============================================================
# Cache directory
# ============================================================

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
# File hashing
# ============================================================

def sha256_file(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# Verify file
# ============================================================

def verify_file(
    path,
    expected_sha256,
):

    if not path.exists():

        return False

    actual_sha256 = sha256_file(
        path
    )

    return (
        actual_sha256
        == expected_sha256
    )


# ============================================================
# Download
# ============================================================

def download_file(
    url,
    destination,
    expected_sha256=None,
):

    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading LexiGround dataset:"
    )

    print(url)

    try:

        urlretrieve(
            url,
            destination,
        )

    except Exception:

        destination.unlink(
            missing_ok=True
        )

        raise

    # --------------------------------------------------------
    # Verify downloaded file
    # --------------------------------------------------------

    if expected_sha256 is not None:

        if not verify_file(
            destination,
            expected_sha256,
        ):

            actual_sha256 = sha256_file(
                destination
            )

            destination.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Downloaded dataset failed "
                "integrity verification.\n"
                f"Expected SHA256: "
                f"{expected_sha256}\n"
                f"Actual SHA256: "
                f"{actual_sha256}"
            )

    print(
        f"Saved to: {destination}"
    )

    return destination


# ============================================================
# Get Lancaster dataset
# ============================================================

def get_lancaster_path():

    cache_dir = get_cache_dir()

    path = (
        cache_dir
        / "lancaster_sensorimotor_norms.csv"
    )

    # --------------------------------------------------------
    # Existing cache
    # --------------------------------------------------------

    if path.exists():

        if verify_file(
            path,
            LANCASTER_SHA256,
        ):

            return path

        # Corrupted or unexpected version
        print(
            "Cached Lancaster dataset failed "
            "integrity verification. "
            "Re-downloading."
        )

        path.unlink(
            missing_ok=True
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    return download_file(
        LANCASTER_URL,
        path,
        expected_sha256=LANCASTER_SHA256,
    )


# ============================================================
# Get Iconicity dataset
# ============================================================

def get_iconicity_path():

    cache_dir = get_cache_dir()

    path = (
        cache_dir
        / "iconicity_ratings_cleaned.csv"
    )

    # --------------------------------------------------------
    # Existing cache
    # --------------------------------------------------------

    if path.exists():

        if verify_file(
            path,
            ICONICITY_SHA256,
        ):

            return path

        # Corrupted or unexpected version
        print(
            "Cached Iconicity dataset failed "
            "integrity verification. "
            "Re-downloading."
        )

        path.unlink(
            missing_ok=True
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    return download_file(
        ICONICITY_URL,
        path,
        expected_sha256=ICONICITY_SHA256,
    )
