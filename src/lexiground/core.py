from pathlib import Path

from .datasets import NormDataset
from .embeddings import create_embedder
from .models import RidgeEstimator


class LexiGround:
    """
    Main public interface for LexiGround.

    LexiGround provides human lexical ratings when available
    and estimates missing ratings using pretrained
    SBERT + Ridge regression models.

    Parameters
    ----------
    lancaster_path : str or Path, optional
        Path to the Lancaster Sensorimotor Norms CSV.

    iconicity_path : str or Path, optional
        Path to the lexical iconicity ratings CSV.

    embedding : str, default="sbert"
        Embedding model used for estimation.

    embedding_kwargs : dict, optional
        Additional arguments passed to the embedding model.
    """

    def __init__(
        self,
        lancaster_path=None,
        iconicity_path=None,
        embedding="sbert",
        embedding_kwargs=None,
    ):

        # -------------------------------------------------
        # DATASETS
        # -------------------------------------------------

        self.datasets = NormDataset(
            lancaster_path=lancaster_path,
            iconicity_path=iconicity_path,
        )

        # -------------------------------------------------
        # EMBEDDING MODEL
        # -------------------------------------------------

        self.embedding_name = embedding

        if embedding_kwargs is None:
            embedding_kwargs = {}

        self.embedder = create_embedder(
            embedding,
            **embedding_kwargs,
        )

        # -------------------------------------------------
        # PRETRAINED MODELS
        # -------------------------------------------------

        self.models = self._load_pretrained_models()

    # =====================================================
    # PRETRAINED MODEL LOADING
    # =====================================================

    def _load_pretrained_models(self):
        """
        Load pretrained SBERT → Ridge models.

        Models are stored in:

            repo/models/sbert/

        Each model is named:

            sbert_<feature>.joblib
        """

        # -------------------------------------------------
        # Development location on BlueBEAR
        # -------------------------------------------------

        model_dir = Path(
            "/rds/projects/p/parkh-speech-linguistics-01/"
            "lexiground/repo/models/sbert"
        )

        models = {}

        if not model_dir.exists():

            print(
                f"Warning: pretrained model directory "
                f"not found:\n{model_dir}"
            )

            return models

        # -------------------------------------------------
        # Load every model
        # -------------------------------------------------

        model_paths = sorted(
            model_dir.glob("sbert_*.joblib")
        )

        for model_path in model_paths:

            feature = (
                model_path.stem
                .replace("sbert_", "")
            )

            model = RidgeEstimator()

            model.load(
                model_path
            )

            models[feature] = model

        print(
            f"Loaded {len(models)} pretrained models."
        )

        return models

    # =====================================================
    # FEATURE INFORMATION
    # =====================================================

    def available_features(self):
        """
        Return features for which human datasets
        are currently available.
        """

        return self.datasets.available_features()

    # =====================================================
    # HUMAN LOOKUP
    # =====================================================

    def lookup_human(
        self,
        word,
        feature,
    ):
        """
        Return the human rating for a word and feature.

        Returns None if the word is not present.
        """

        return self.datasets.lookup(
            word,
            feature,
        )

    # =====================================================
    # MODEL CHECK
    # =====================================================

    def _ensure_model(
        self,
        feature,
    ):
        """
        Check whether a pretrained model exists.
        """

        if feature not in self.models:

            raise ValueError(
                f"No pretrained model is available "
                f"for feature '{feature}'. "
                f"Available models: "
                f"{list(self.models.keys())}"
            )

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(
        self,
        word,
        feature,
    ):
        """
        Estimate a lexical rating using:

            word
              ↓
            SBERT
              ↓
        pretrained Ridge
              ↓
          prediction
        """

        self._ensure_model(
            feature
        )

        embedding = self.embedder.encode(
            [str(word)]
        )

        prediction = self.models[
            feature
        ].predict(
            embedding
        )[0]

        return float(
            prediction
        )

    # =====================================================
    # SINGLE FEATURE LOOKUP
    # =====================================================

    def lookup(
        self,
        word,
        feature,
        estimate_missing=True,
    ):
        """
        Return either a human rating or an estimate.

        Returns
        -------
        dict
            Contains:

            word
            feature
            value
            source
        """

        # -------------------------------------------------
        # Try human rating first
        # -------------------------------------------------

        human = self.lookup_human(
            word,
            feature,
        )

        if human is not None:

            return {
                "word": str(word),
                "feature": feature,
                "value": human,
                "source": "human",
            }

        # -------------------------------------------------
        # Word missing and estimation disabled
        # -------------------------------------------------

        if not estimate_missing:

            return {
                "word": str(word),
                "feature": feature,
                "value": None,
                "source": "missing",
            }

        # -------------------------------------------------
        # Estimate using pretrained model
        # -------------------------------------------------

        prediction = self.predict(
            word,
            feature,
        )

        return {
            "word": str(word),
            "feature": feature,
            "value": prediction,
            "source": "estimated",
            "embedding": self.embedding_name,
        }

    # =====================================================
    # MAIN USER API
    # =====================================================

    def get(
        self,
        word,
    ):
        """
        Return all available lexical ratings for a word.

        Human ratings are used when available.

        Otherwise, pretrained SBERT + Ridge models
        are used to estimate the rating.
        """

        word = str(word).strip()

        results = {
            "word": word,
            "features": {},
        }

        # -------------------------------------------------
        # We use the features for which we have
        # pretrained models.
        # -------------------------------------------------

        for feature in self.models.keys():

            results["features"][feature] = (
                self.lookup(
                    word,
                    feature,
                    estimate_missing=True,
                )
            )

        return results
