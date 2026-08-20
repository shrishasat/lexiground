import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def validate_predictions(
    y_true,
    y_pred,
):
    """
    Calculate validation metrics for predicted
    lexical ratings.

    Parameters
    ----------
    y_true : array-like
        Human ratings.

    y_pred : array-like
        Model predictions.

    Returns
    -------
    dict
        Pearson r, Spearman rho, R2, RMSE and MAE.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    # -------------------------------------------------
    # Remove invalid values
    # -------------------------------------------------

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) < 2:

        raise ValueError(
            "At least two valid observations "
            "are required."
        )

    # -------------------------------------------------
    # Pearson correlation
    # -------------------------------------------------

    pearson_r, pearson_p = pearsonr(
        y_true,
        y_pred,
    )

    # -------------------------------------------------
    # Spearman correlation
    # -------------------------------------------------

    spearman_rho, spearman_p = spearmanr(
        y_true,
        y_pred,
    )

    # -------------------------------------------------
    # R2
    # -------------------------------------------------

    r2 = r2_score(
        y_true,
        y_pred,
    )

    # -------------------------------------------------
    # RMSE
    # -------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    # -------------------------------------------------
    # MAE
    # -------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    return {
        "n": len(y_true),
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_rho": spearman_rho,
        "Spearman_p": spearman_p,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
    }


def validate_feature(
    dataset,
    word_column,
    rating_column,
    model,
    embedder,
):
    """
    Validate one pretrained model against
    human ratings.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Human norm dataset.

    word_column : str
        Column containing words.

    rating_column : str
        Column containing human ratings.

    model : RidgeEstimator
        Pretrained model.

    embedder : embedding model
        SBERT embedder.

    Returns
    -------
    dict
        Validation metrics.
    """

    valid = (
        dataset[word_column].notna()
        & dataset[rating_column].notna()
    )

    words = (
        dataset.loc[
            valid,
            word_column,
        ]
        .astype(str)
        .str.strip()
    )

    ratings = (
        dataset.loc[
            valid,
            rating_column,
        ]
        .astype(float)
    )

    # Remove empty words
    nonempty = words != ""

    words = words.loc[
        nonempty
    ]

    ratings = ratings.loc[
        words.index
    ]

    print(
        f"Encoding {len(words):,} words..."
    )

    embeddings = embedder.encode(
        words.tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    predictions = model.predict(
        embeddings
    )

    metrics = validate_predictions(
        ratings.to_numpy(),
        predictions,
    )

    return metrics


def validation_table(results):
    """
    Convert validation results into a DataFrame.

    Parameters
    ----------
    results : dict
        Mapping from feature name to metrics.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for feature, metrics in results.items():

        row = {
            "Feature": feature,
            **metrics,
        }

        rows.append(row)

    return pd.DataFrame(
        rows
    )
