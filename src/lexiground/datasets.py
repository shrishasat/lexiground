from pathlib import Path

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
    """
    Load and access lexical norm datasets.

    Lancaster Sensorimotor Norms:
        Lynott et al. (2019)

    Iconicity ratings:
        User-provided iconicity norm dataset.
    """

    def __init__(
        self,
        lancaster_path=None,
        iconicity_path=None,
    ):

        self.lancaster = None
        self.iconicity = None

        if lancaster_path is not None:
            self.lancaster = self._load_lancaster(
                lancaster_path
            )

        if iconicity_path is not None:
            self.iconicity = self._load_iconicity(
                iconicity_path
            )

    @staticmethod
    def _check_path(path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        return path

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

        # Standardise word column
        data = data.copy()

        data["Word"] = (
            data["Word"]
            .astype(str)
            .str.strip()
        )

        return data

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

        # Standardise iconicity word column
        data["word"] = (
            data["word"]
            .astype(str)
            .str.strip()
        )

        return data

    def get_lancaster(self):

        if self.lancaster is None:
            raise ValueError(
                "Lancaster dataset was not loaded."
            )

        return self.lancaster

    def get_iconicity(self):

        if self.iconicity is None:
            raise ValueError(
                "Iconicity dataset was not loaded."
            )

        return self.iconicity

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

    def lookup(self, word, feature):

        word = str(word).strip().lower()

        # Lancaster
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

        # Iconicity
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
