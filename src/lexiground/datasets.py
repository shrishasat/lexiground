from pathlib import Path
import os
import pandas as pd


LANCASTER_FEATURES = {
    "Auditory": "Auditory.mean",
    "Gustatory": "Gustatory.mean",
    "Haptic": "Haptic.mean",
    "Interoceptive": "Interoceptive.mean",
    "Olfactory": "Olfactory.mean",
    "Visual": "Visual.mean",
    "Foot_leg": "Foot_leg.mean",
    "Hand_arm": "Hand_arm.mean",
    "Head": "Head.mean",
    "Mouth": "Mouth.mean",
    "Torso": "Torso.mean",
    "Minkowski3_perceptual": "Minkowski3.perceptual",
    "Minkowski3_action": "Minkowski3.action",
    "Minkowski3_sensorimotor": "Minkowski3.sensorimotor",
}

ICONICITY_FEATURES = {
    "Iconicity": "rating",
}


class NormDataset:

    def __init__(
        self,
        lancaster_path=None,
        iconicity_path=None,
        cache_dir=None,
    ):

        self.lancaster = None
        self.iconicity = None

        # --------------------------------------------------
        # Cache directory
        # --------------------------------------------------

        if cache_dir is None:

            cache_dir = (
                Path.home()
                / ".cache"
                / "lexiground"
            )

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Lancaster
        # --------------------------------------------------

        if lancaster_path is not None:

            self.lancaster = self._load_lancaster(
                lancaster_path
            )

        # --------------------------------------------------
        # Iconicity
        # --------------------------------------------------

        if iconicity_path is not None:

            self.iconicity = self._load_iconicity(
                iconicity_path
            )

    # ======================================================
    # PATH HANDLING
    # ======================================================

    @staticmethod
    def _check_path(path):

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        return path

    # ======================================================
    # LOAD LANCASTER
    # ======================================================

    @classmethod
    def _load_lancaster(cls, path):

        path = cls._check_path(path)

        data = pd.read_csv(path)

        required_columns = [
            "Word",
            *LANCASTER_FEATURES.values(),
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Lancaster dataset is missing "
                f"required columns: {missing}"
            )

        data = data.copy()

        data["Word"] = (
            data["Word"]
            .astype(str)
            .str.strip()
        )

        return data

    # ======================================================
    # LOAD ICONICITY
    # ======================================================

    @classmethod
    def _load_iconicity(cls, path):

        path = cls._check_path(path)

        data = pd.read_csv(path)

        required_columns = [
            "word",
            "rating",
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Iconicity dataset is missing "
                f"required columns: {missing}"
            )

        data = data.copy()

        data["word"] = (
            data["word"]
            .astype(str)
            .str.strip()
        )

        return data

    # ======================================================
    # DATA ACCESS
    # ======================================================

    def get_lancaster(self):

        if self.lancaster is None:

            raise ValueError(
                "Lancaster dataset is not available."
            )

        return self.lancaster

    def get_iconicity(self):

        if self.iconicity is None:

            raise ValueError(
                "Iconicity dataset is not available."
            )

        return self.iconicity

    # ======================================================
    # FEATURES
    # ======================================================

    def available_lancaster_features(self):

        return list(
            LANCASTER_FEATURES.keys()
        )

    def available_features(self):

        features = []

        if self.lancaster is not None:

            features.extend(
                LANCASTER_FEATURES.keys()
            )

        if self.iconicity is not None:

            features.extend(
                ICONICITY_FEATURES.keys()
            )

        return features

    # ======================================================
    # LOOKUP
    # ======================================================

    def lookup(
        self,
        word,
        feature,
    ):

        word = (
            str(word)
            .strip()
            .lower()
        )

        # --------------------------------------------------
        # Lancaster
        # --------------------------------------------------

        if feature in LANCASTER_FEATURES:

            data = self.get_lancaster()

            matches = data[
                data["Word"]
                .str.lower()
                == word
            ]

            if matches.empty:

                return None

            column = LANCASTER_FEATURES[
                feature
            ]

            value = matches.iloc[0][column]

            if pd.isna(value):

                return None

            return float(value)

        # --------------------------------------------------
        # Iconicity
        # --------------------------------------------------

        if feature in ICONICITY_FEATURES:

            data = self.get_iconicity()

            matches = data[
                data["word"]
                .str.lower()
                == word
            ]

            if matches.empty:

                return None

            column = ICONICITY_FEATURES[
                feature
            ]

            value = matches.iloc[0][column]

            if pd.isna(value):

                return None

            return float(value)

        raise ValueError(
            f"Unknown feature: {feature}. "
            f"Available features: "
            f"{self.available_features()}"
        )
