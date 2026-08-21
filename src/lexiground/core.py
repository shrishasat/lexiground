from pathlib import Path

from .datasets import NormDataset
from .embeddings import create_embedder
from .models import RidgeEstimator


class LexiGround:
    """
    Main public interface for LexiGround.

    LexiGround returns human lexical norm ratings when available.
    For words without human ratings, pretrained SBERT + Ridge models
    are used to estimate the missing values.

    Users normally only need:

        from lexiground import LexiGround

        lex = LexiGround()

        result = lex.get("word")
    """

    def __init__(
        self,
        lancaster_path=None,
        iconicity_path=None,
        embedding="sbert",
        embedding_kwargs=None,
    ):

        # ---------------------------------------------------------
        # DATASETS
        # ---------------------------------------------------------

        self.datasets = NormDataset(
            lancaster_path=lancaster_path,
            iconicity_path=iconicity_path,
        )

        # ---------------------------------------------------------
        # EMBEDDING MODEL
        # ---------------------------------------------------------

        self.embedding_name = embedding

        if embedding_kwargs is None:
            embedding_kwargs = {}

        self.embedder = create_embedder(
            embedding,
            **embedding_kwargs,
        )

        # ---------------------------------------------------------
        # PRETRAINED MODELS
        # ---------------------------------------------------------

        self.models = self._load_pretrained_models()

    # =============================================================
    # DATASET INFORMATION
    # =============================================================

    def available_features(self):
        """
        Return all available lexical features.
        """

        return self.datasets.available_features()

    # =============================================================
    # HUMAN LOOKUP
    # =============================================================

    def lookup_human(
        self,
        word,
        feature,
    ):
        """
        Return the human rating for a word and feature.

        Returns None if the word does not have a human rating.
        """

        return self.datasets.lookup(
            word,
            feature,
        )

    # =============================================================
    # PRETRAINED MODEL LOADING
    # =============================================================

    def _load_pretrained_models(self):
        """
        Load pretrained SBERT → Ridge models bundled with
        the LexiGround package.

        Models are stored inside:

            lexiground/models/sbert/
        """

        model_dir = (
            Path(__file__).resolve()
            .parent
            / "models"
            / "sbert"
        )

        models = {}

        if not model_dir.exists():
            return models

        for model_path in model_dir.glob(
            "sbert_*.joblib"
        ):

            feature = (
                model_path.stem
                .replace("sbert_", "")
            )

            model = RidgeEstimator()

            model.load(
                model_path
            )

            models[feature] = model

        return models

    # =============================================================
    # MODEL AVAILABILITY
    # =============================================================

    def _ensure_model(
        self,
        feature,
    ):
        """
        Check that a pretrained model exists for the feature.
        """

        if feature not in self.models:

            raise ValueError(
                f"No pre-trained model is available "
                f"for feature '{feature}'. "
                f"Available model features: "
                f"{list(self.models.keys())}"
            )

    # =============================================================
    # PREDICTION
    # =============================================================

    def predict(
        self,
        word,
        feature,
    ):
        """
        Estimate a lexical rating using a pretrained
        SBERT + Ridge model.

        Parameters
        ----------
        word : str
            Word to estimate.

        feature : str
            Lexical feature to estimate.

        Returns
        -------
        float
            Predicted lexical rating.
        """

        self._ensure_model(
            feature
        )

        X = self.embedder.encode(
            [str(word)]
        )

        prediction = self.models[
            feature
        ].predict(X)[0]

        return float(
            prediction
        )

    # =============================================================
    # SINGLE FEATURE LOOKUP
    # =============================================================

    def lookup(
        self,
        word,
        feature,
        estimate_missing=True,
    ):
        """
        Return the human or estimated rating for one feature.

        Human ratings take priority. If no human rating exists,
        the pretrained SBERT + Ridge model is used automatically.
        """

        word = str(word).strip()

        # ---------------------------------------------------------
        # Try human rating first
        # ---------------------------------------------------------

        human = self.lookup_human(
            word,
            feature,
        )

        if human is not None:

            return {
                "word": word,
                "feature": feature,
                "value": human,
                "source": "human",
            }

        # ---------------------------------------------------------
        # Return missing if estimation disabled
        # ---------------------------------------------------------

        if not estimate_missing:

            return {
                "word": word,
                "feature": feature,
                "value": None,
                "source": "missing",
            }

        # ---------------------------------------------------------
        # Estimate using pretrained model
        # ---------------------------------------------------------

        prediction = self.predict(
            word,
            feature,
        )

        return {
            "word": word,
            "feature": feature,
            "value": prediction,
            "source": "estimated",
            "embedding": self.embedding_name,
        }

    # =============================================================
    # MAIN USER API
    # =============================================================

    def get(
        self,
        word,
    ):
        """
        Return all lexical features for a word.

        Human ratings are returned where available.
        Missing ratings are automatically estimated using
        pretrained SBERT + Ridge models.

        Parameters
        ----------
        word : str
            Word to look up.

        Returns
        -------
        dict
            Dictionary containing the word and all lexical features.

        Example
        -------
        >>> lex = LexiGround()
        >>> lex.get("neuroscience")
        """

        word = str(word).strip()

        results = {
            "word": word,
            "features": {},
        }

        for feature in self.available_features():

            results["features"][feature] = (
                self.lookup(
                    word,
                    feature,
                    estimate_missing=True,
                )
            )

        return results
