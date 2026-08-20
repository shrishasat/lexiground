import joblib

from sklearn.linear_model import Ridge


class RidgeEstimator:
    """
    Ridge regression model for estimating lexical ratings
    from word embeddings.
    """

    def __init__(self, alpha=1.0):

        self.alpha = alpha
        self.model = Ridge(
            alpha=alpha
        )

    def fit(self, X, y):

        self.model.fit(X, y)

        return self

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        joblib.dump(
            self.model,
            path,
        )

    def load(self, path):

        self.model = joblib.load(
            path
        )

        return self
