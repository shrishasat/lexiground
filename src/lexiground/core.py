import numpy as np

from .datasets import NormDataset
from .embeddings import create_embedder
from .models import RidgeEstimator


class LexiGround:

    def __init__(
        self,
        lancaster_path=None,
        iconicity_path=None,
        embedding="sbert",
    ):

        self.datasets = NormDataset(
            lancaster_path=lancaster_path,
            iconicity_path=iconicity_path,
        )

        self.embedding_name = embedding

        self.embedder = create_embedder(
            embedding
        )

        self.models = {}

    def lookup_human(
        self,
        word,
        dataset="lancaster",
    ):

        if dataset == "lancaster":
            data = self.datasets.get_lancaster()

        elif dataset == "iconicity":
            data = self.datasets.get_iconicity()

        else:
            raise ValueError(
                "dataset must be 'lancaster' "
                "or 'iconicity'"
            )

        matches = data[
            data["Word"]
            .astype(str)
            .str.lower()
            == word.lower()
        ]

        if len(matches) == 0:
            return None

        return matches.iloc[0].to_dict()

    def lookup(
        self,
        word,
        dataset="lancaster",
        feature=None,
    ):

        human = self.lookup_human(
            word,
            dataset=dataset,
        )

        if human is not None:

            if feature is None:
                return {
                    "word": word,
                    "source": "human",
                    "values": human,
                }

            if feature in human:
                return {
                    "word": word,
                    "feature": feature,
                    "value": human[feature],
                    "source": "human",
                }

        if feature is None:
            raise ValueError(
                "Word not found and no feature was specified "
                "for estimation."
            )

        return self.predict(
            word,
            dataset=dataset,
            feature=feature,
        )

    def fit(
        self,
        dataset,
        feature,
    ):

        if dataset == "lancaster":
            data = self.datasets.get_lancaster()

        elif dataset == "iconicity":
            data = self.datasets.get_iconicity()

        else:
            raise ValueError(
                "Unknown dataset."
            )

        words = data["Word"].astype(str)

        valid = data[feature].notna()

        words = words[valid].tolist()

        y = data.loc[
            valid,
            feature
        ].to_numpy()

        X = self.embedder.encode(words)

        model = RidgeEstimator()

        model.fit(X, y)

        self.models[
            (dataset, feature)
        ] = model

        return self

    def predict(
        self,
        word,
        dataset,
        feature,
    ):

        key = (dataset, feature)

        if key not in self.models:
            raise ValueError(
                f"Model for {dataset}/{feature} "
                "has not been fitted."
            )

        embedding = self.embedder.encode(
            [word]
        )

        prediction = self.models[
            key
        ].predict(embedding)[0]

        return {
            "word": word,
            "dataset": dataset,
            "feature": feature,
            "value": float(prediction),
            "source": "estimated",
            "embedding": self.embedding_name,
        }
