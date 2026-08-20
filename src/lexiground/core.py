from .datasets import NormDataset
from .embeddings import create_embedder
from .models import RidgeEstimator


class LexiGround:
    """
    Main public interface for LexiGround.

    For each word, LexiGround returns human ratings when available
    and SBERT/Ridge estimates when a human rating is unavailable.
    """

    def __init__(
        self,
        lancaster_path=None,
        iconicity_path=None,
        embedding="sbert",
        embedding_kwargs=None,
    ):

        self.datasets = NormDataset(
            lancaster_path=lancaster_path,
            iconicity_path=iconicity_path,
        )

        self.embedding_name = embedding

        if embedding_kwargs is None:
            embedding_kwargs = {}

        self.embedder = create_embedder(
            embedding,
            **embedding_kwargs,
        )

        # Models are fitted lazily.
        # This means the user does not have to call fit().
        self.models = {}

    # ---------------------------------------------------------
    # DATASET INFORMATION
    # ---------------------------------------------------------

    def available_features(self):
        """Return all available lexical features."""
        return self.datasets.available_features()

    # ---------------------------------------------------------
    # HUMAN LOOKUP
    # ---------------------------------------------------------

    def lookup_human(self, word, feature):
        """Return the human rating if available."""
        return self.datasets.lookup(
            word,
            feature,
        )

    # ---------------------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------------------

    def _fit_feature(self, feature, alpha=1.0):
        """
        Internally fit an SBERT → Ridge model for one feature.

        Users normally do not need to call this.
        """

        if feature in self.datasets.available_lancaster_features():

            data = self.datasets.get_lancaster()

            column_map = {
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
                "Minkowski3_perceptual":
                    "Minkowski3.perceptual",
                "Minkowski3_action":
                    "Minkowski3.action",
                "Minkowski3_sensorimotor":
                    "Minkowski3.sensorimotor",
            }

            word_column = "Word"
            rating_column = column_map[feature]

        elif feature == "Iconicity":

            data = self.datasets.get_iconicity()

            word_column = "word"
            rating_column = "rating"

        else:

            raise ValueError(
                f"Unknown feature: {feature}"
            )

        valid = data[rating_column].notna()

        words = (
            data.loc[
                valid,
                word_column
            ]
            .astype(str)
            .tolist()
        )

        y = data.loc[
            valid,
            rating_column
        ].to_numpy()

        # Generate SBERT embeddings
        X = self.embedder.encode(words)

        # Train Ridge
        model = RidgeEstimator(
            alpha=alpha
        )

        model.fit(
            X,
            y
        )

        self.models[feature] = model

    def _ensure_model(self, feature):
        """
        Make sure an estimator exists for the requested feature.

        If it has not yet been trained, train it automatically.
        """

        if feature not in self.models:
            self._fit_feature(feature)

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    def predict(self, word, feature):
        """
        Estimate a lexical rating using SBERT + Ridge.
        """

        self._ensure_model(feature)

        X = self.embedder.encode(
            [word]
        )

        prediction = self.models[
            feature
        ].predict(X)[0]

        return float(prediction)

    # ---------------------------------------------------------
    # SINGLE FEATURE LOOKUP
    # ---------------------------------------------------------

    def lookup(
        self,
        word,
        feature,
        estimate_missing=True,
    ):
        """
        Return human or estimated rating for one feature.
        """

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

        if not estimate_missing:

            return {
                "word": word,
                "feature": feature,
                "value": None,
                "source": "missing",
            }

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

    # ---------------------------------------------------------
    # MAIN USER API
    # ---------------------------------------------------------

    def get(self, word):
        """
        Return all lexical features for a word.

        Human ratings are returned where available.
        Missing ratings are estimated automatically.
        """

        results = {
            "word": str(word),
            "features": {},
        }

        for feature in self.available_features():

            results["features"][feature] = self.lookup(
                word,
                feature,
                estimate_missing=True,
            )

        return results
