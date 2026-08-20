from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SBERT_MODEL = "all-mpnet-base-v2"
RIDGE_ALPHA = 1.0

LANCASTER_PATH = Path(
    "/rds/projects/p/parkh-speech-linguistics-01/"
    "Final_results/Lancaster_Sensorimotor_Norms/"
    "Lancaster_Sensorimotor_40k_database.csv"
)

ICONICITY_PATH = Path(
    "/rds/projects/p/parkh-speech-linguistics-01/"
    "TrainingRatedData/iconicity_ratings.csv"
)

OUTPUT_DIR = Path(
    "/rds/projects/p/parkh-speech-linguistics-01/"
    "lexiground/repo/models/sbert"
)


# ---------------------------------------------------------
# FEATURE DEFINITIONS
# ---------------------------------------------------------

LANCASTER_FEATURES = {
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


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_datasets():

    print("Loading Lancaster...")

    lancaster = pd.read_csv(
        LANCASTER_PATH
    )

    print(
        f"Lancaster words: {len(lancaster):,}"
    )

    print("Loading Iconicity...")

    iconicity = pd.read_csv(
        ICONICITY_PATH
    )

    print(
        f"Iconicity words: {len(iconicity):,}"
    )

    return lancaster, iconicity


# ---------------------------------------------------------
# TRAIN ONE MODEL
# ---------------------------------------------------------

def train_model(
    words,
    ratings,
    embedder,
    feature,
):

    print(
        f"\nTraining SBERT model: {feature}"
    )

    valid = ratings.notna()

    words = (
        words.loc[valid]
        .astype(str)
        .str.strip()
        .tolist()
    )

    y = ratings.loc[valid].to_numpy()

    print(
        f"Training examples: {len(words):,}"
    )

    print("Generating SBERT embeddings...")

    X = embedder.encode(
        words,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(
        f"Embedding shape: {X.shape}"
    )

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X,
        y,
    )

    output_path = (
        OUTPUT_DIR
        / f"sbert_{feature}.joblib"
    )

    joblib.dump(
        model,
        output_path,
    )

    print(
        f"Saved: {output_path}"
    )

    return model


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lancaster, iconicity = load_datasets()

    print(
        f"\nLoading SBERT: {SBERT_MODEL}"
    )

    embedder = SentenceTransformer(
        SBERT_MODEL
    )

    # -----------------------------------------------------
    # LANCASTER MODELS
    # -----------------------------------------------------

    for feature, column in LANCASTER_FEATURES.items():

        train_model(
            words=lancaster["Word"],
            ratings=lancaster[column],
            embedder=embedder,
            feature=feature,
        )

    # -----------------------------------------------------
    # ICONICITY MODEL
    # -----------------------------------------------------

    train_model(
        words=iconicity["word"],
        ratings=iconicity["rating"],
        embedder=embedder,
        feature="Iconicity",
    )

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED")
    print("=" * 60)

    models = sorted(
        OUTPUT_DIR.glob(
            "sbert_*.joblib"
        )
    )

    for model in models:
        print(model.name)

    print(
        f"\nTotal models: {len(models)}"
    )


if __name__ == "__main__":
    main()
