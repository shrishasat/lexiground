import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold


def evaluate_predictions(y_true, y_pred):

    r, p = pearsonr(y_true, y_pred)

    return {
        "Pearson_r": r,
        "Pearson_p": p,
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),
    }


def cross_validate_embedding(
    embeddings,
    ratings,
    alpha=1.0,
    n_splits=5,
):

    from sklearn.linear_model import Ridge

    embeddings = np.asarray(embeddings)
    ratings = np.asarray(ratings)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    predictions = np.zeros_like(
        ratings,
        dtype=float,
    )

    for train_idx, test_idx in kfold.split(embeddings):

        model = Ridge(alpha=alpha)

        model.fit(
            embeddings[train_idx],
            ratings[train_idx],
        )

        predictions[test_idx] = model.predict(
            embeddings[test_idx]
        )

    metrics = evaluate_predictions(
        ratings,
        predictions,
    )

    return metrics, predictions
