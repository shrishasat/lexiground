import numpy as np


class BaseEmbedder:
    """Base interface for word embedding models."""

    def encode(self, words):
        raise NotImplementedError


class SBERTEmbedder(BaseEmbedder):
    """
    Sentence-BERT embedding model.

    Default model:
        all-mpnet-base-v2
    """

    def __init__(
        self,
        model_name="all-mpnet-base-v2",
    ):

        from sentence_transformers import (
            SentenceTransformer
        )

        self.model = SentenceTransformer(
            model_name
        )

    def encode(self, words):

        return self.model.encode(
            words,
            convert_to_numpy=True,
            show_progress_bar=True,
        )


class GloVeEmbedder(BaseEmbedder):
    """
    Wrapper for a loaded GloVe embedding model.
    """

    def __init__(self, vectors):

        self.vectors = vectors

    def encode(self, words):

        embeddings = []

        for word in words:

            if word in self.vectors:

                embeddings.append(
                    self.vectors[word]
                )

            else:

                embeddings.append(
                    np.zeros(
                        self.vectors.vector_size
                    )
                )

        return np.asarray(
            embeddings
        )


class FastTextEmbedder(BaseEmbedder):
    """
    Wrapper for a loaded FastText model.

    FastText can generate vectors for
    out-of-vocabulary words using subword
    information.
    """

    def __init__(self, model):

        self.model = model

    def encode(self, words):

        return np.asarray([
            self.model.get_word_vector(word)
            for word in words
        ])


def create_embedder(
    name,
    **kwargs,
):

    name = name.lower()

    if name == "sbert":

        return SBERTEmbedder(
            **kwargs
        )

    if name == "glove":

        return GloVeEmbedder(
            **kwargs
        )

    if name == "fasttext":

        return FastTextEmbedder(
            **kwargs
        )

    raise ValueError(
        f"Unknown embedding model: {name}. "
        "Choose from: sbert, glove, fasttext."
    )
