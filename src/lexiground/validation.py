import numpy as np

from scipy.stats import pearsonr

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


def evaluate_predictions(
    y_true,
    y_pred,
):

    r, p = pearsonr(
        y_true,
        y_pred,
    )

    return {
        "Pearson_r": float(r),
        "Pearson_p": float(p),
        "R2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
    }


def cross_validate(
    X,
    y,
    alpha=1.0,
    n_splits=5,
):

    X = np.asarray(X)
    y = np.asarray(y)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    predictions = np.zeros(
        len(y)
    )

    for train_idx, test_idx in kfold.split(X):

        model = Ridge(
            alpha=alpha
        )

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        predictions[test_idx] = (
            model.predict(
                X[test_idx]
            )
        )

    metrics = evaluate_predictions(
        y,
        predictions,
    )

    return metrics, predictions
