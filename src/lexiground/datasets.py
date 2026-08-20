from pathlib import Path
import pandas as pd


class NormDataset:
    """Load and manage lexical norm datasets."""

    def __init__(self, lancaster_path=None, iconicity_path=None):
        self.lancaster = None
        self.iconicity = None

        if lancaster_path is not None:
            self.lancaster = self._load_excel(lancaster_path)

        if iconicity_path is not None:
            self.iconicity = self._load_excel(iconicity_path)

    @staticmethod
    def _load_excel(path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        raise ValueError(
            "Supported formats are .xlsx, .xls and .csv"
        )

    def get_lancaster(self):
        if self.lancaster is None:
            raise ValueError("Lancaster dataset was not loaded.")
        return self.lancaster

    def get_iconicity(self):
        if self.iconicity is None:
            raise ValueError("Iconicity dataset was not loaded.")
        return self.iconicity
