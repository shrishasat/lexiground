from .datasets import NormDataset
from .embeddings import create_embedder
from .models import RidgeEstimator


class LexiGround:
    """
    Main LexiGround interface.

    Provides human lexical norm lookup and
    model-based estimation for missing words.
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

        self.models = {}

    def available_features(self):

        return self.datasets.available_features()

    def lookup_human(
        self,
        word,
        feature,
    ):

        return self.datasets.lookup(
            word,
            feature,
        )

    def fit(
        self,
        feature,
        alpha=1.0,
    ):

        # Determine which dataset contains feature

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

        valid = data[
            rating_column
        ].notna()

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

        X = self.embedder.encode(
            words
        )

        model = RidgeEstimator(
            alpha=alpha
        )

        model.fit(
            X,
            y
        )

        self.models[feature] = model

        return self

    def predict(
        self,
        word,
        feature,
    ):

        if feature not in self.models:

            raise ValueError(
                f"No model has been fitted for "
                f"{feature}. Run .fit('{feature}') first."
            )

        X = self.embedder.encode(
            [word]
        )

        prediction = self.models[
            feature
        ].predict(X)[0]

        return float(prediction)

    def lookup(
        self,
        word,
        feature,
        estimate_missing=True,
    ):

        # First try human rating
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

        # Missing word
        if not estimate_missing:

            return {
                "word": word,
                "feature": feature,
                "value": None,
                "source": "missing",
            }

        # Estimate if model exists
        if feature not in self.models:

            raise ValueError(
                f"'{word}' was not found in the "
                f"human norms and no estimation model "
                f"has been fitted for '{feature}'. "
                f"Run lex.fit('{feature}') first."
            )

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
