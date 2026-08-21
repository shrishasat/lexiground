from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from lexiground.datasets import (
    LANCASTER_FEATURES,
    NormDataset,
)

from lexiground.embeddings import create_embedder
from lexiground.models import RidgeEstimator

from lexiground.validation import validate_predictions


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
ALPHA = 1.0

EMBEDDING = "sbert"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET PATHS
# ============================================================

LANCASTER_PATH = (
    "/rds/projects/p/parkh-speech-linguistics-01/"
    "Final_results/Lancaster_Sensorimotor_Norms/"
    "Lancaster_Sensorimotor_40k_database.csv"
)

ICONICITY_PATH = (
    "/rds/projects/p/parkh-speech-linguistics-01/"
    "TrainingRatedData/"
    "iconicity_ratings.csv"
)


# ============================================================
# FEATURE → COLUMN MAPPING
# ============================================================

FEATURE_COLUMNS = {
    **LANCASTER_FEATURES,
    "Iconicity": "rating",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_feature_data(
    feature,
    lancaster,
    iconicity,
):

    if feature in LANCASTER_FEATURES:

        words = lancaster["Word"]

        ratings = lancaster[
            LANCASTER_FEATURES[feature]
        ]

    elif feature == "Iconicity":

        words = iconicity["word"]

        ratings = iconicity["rating"]

    else:

        raise ValueError(
            f"Unknown feature: {feature}"
        )

    data = pd.DataFrame({
        "word": words,
        "rating": ratings,
    })

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "word",
            "rating",
        ]
    )

    # --------------------------------------------------------
    # Standardise words
    # --------------------------------------------------------

    data["word"] = (
        data["word"]
        .astype(str)
        .str.strip()
    )

    # Remove empty words

    data = data[
        data["word"] != ""
    ]

    # Remove duplicate words
    #
    # Important: we don't want the same lexical item
    # appearing multiple times across train/test.

    data = data.drop_duplicates(
        subset="word"
    )

    return data.reset_index(
        drop=True
    )


# ============================================================
# VALIDATE ONE FEATURE
# ============================================================

def validate_feature(
    feature,
    data,
    embedder,
):

    print()
    print("=" * 70)
    print(
        f"VALIDATING: {feature}"
    )
    print("=" * 70)

    print(
        f"Total examples: {len(data):,}"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training examples: {len(train_data):,}"
    )

    print(
        f"Test examples: {len(test_data):,}"
    )

    # --------------------------------------------------------
    # Generate training embeddings
    # --------------------------------------------------------

    print(
        "Generating training SBERT embeddings..."
    )

    X_train = embedder.encode(
        train_data["word"].tolist()
    )

    y_train = (
        train_data["rating"]
        .to_numpy(dtype=float)
    )

    print(
        f"Training embedding shape: "
        f"{X_train.shape}"
    )

    # --------------------------------------------------------
    # Train temporary Ridge model
    # --------------------------------------------------------

    model = RidgeEstimator(
        alpha=ALPHA
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Generate TEST embeddings
    # --------------------------------------------------------

    print(
        "Generating test SBERT embeddings..."
    )

    X_test = embedder.encode(
        test_data["word"].tolist()
    )

    y_test = (
        test_data["rating"]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Predict unseen words
    # --------------------------------------------------------

    print(
        "Predicting held-out words..."
    )

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = validate_predictions(
        y_test,
        y_pred,
    )

    # Add feature information

    metrics["Feature"] = feature

    # Put Feature first

    metrics = {
        "Feature": metrics.pop("Feature"),
        **metrics,
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        f"Results for {feature}"
    )

    print(
        f"n = {metrics['n']:,}"
    )

    print(
        f"Pearson r = "
        f"{metrics['Pearson_r']:.3f}"
    )

    print(
        f"Pearson p = "
        f"{metrics['Pearson_p']:.4g}"
    )

    print(
        f"Spearman rho = "
        f"{metrics['Spearman_rho']:.3f}"
    )

    print(
        f"Spearman p = "
        f"{metrics['Spearman_p']:.4g}"
    )

    print(
        f"R² = "
        f"{metrics['R2']:.3f}"
    )

    print(
        f"RMSE = "
        f"{metrics['RMSE']:.3f}"
    )

    print(
        f"MAE = "
        f"{metrics['MAE']:.3f}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LexiGround SBERT Validation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    datasets = NormDataset(
        lancaster_path=LANCASTER_PATH,
        iconicity_path=ICONICITY_PATH,
    )

    lancaster = datasets.get_lancaster()

    iconicity = datasets.get_iconicity()

    print(
        f"Lancaster words: "
        f"{len(lancaster):,}"
    )

    print(
        f"Iconicity words: "
        f"{len(iconicity):,}"
    )

    # --------------------------------------------------------
    # Create SBERT embedder
    # --------------------------------------------------------

    print()
    print("Loading SBERT...")

    embedder = create_embedder(
        EMBEDDING
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = list(
        LANCASTER_FEATURES.keys()
    )

    features.append(
        "Iconicity"
    )

    # --------------------------------------------------------
    # Validate every feature
    # --------------------------------------------------------

    all_results = []

    for feature in features:

        data = load_feature_data(
            feature=feature,
            lancaster=lancaster,
            iconicity=iconicity,
        )

        result = validate_feature(
            feature=feature,
            data=data,
            embedder=embedder,
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    output_file = (
        RESULTS_DIR
        / "sbert_validation.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved validation results to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
