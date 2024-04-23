import csv
import json
import logging
import pathlib
from functools import partial
from multiprocessing import Pool
from typing import List, Type

import joblib
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from vectorizer import ImageVectorizer

# These paths are the same as in `../scripts/generate-subimages.py`.
ARTIFACTS_DIR = pathlib.Path("../artifacts")
DATA_DIR = pathlib.Path("../data")
OUT_DIR = DATA_DIR / "subdata/"

# Reproducibility from known data.
RANDOM_STATE = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TRAIN")


def load_features(sub_img_filename, n_color_slices: int = 100, sat_cut: float = 0.65):
    logger.debug(f"loading features for {sub_img_filename}")
    return ImageVectorizer.extract_features(
        str(OUT_DIR / sub_img_filename), n_color_slices, sat_cut
    ).tolist()


class Trainer:
    def __init__(
        self,
        features_train: List[List[float]],
        labels_train: List[str],
        features_test: List[List[float]],
        labels_test: List[str],
        ModelClass: Type[ImageVectorizer | KNeighborsRegressor],
    ):
        self.features_train = np.array(features_train)
        self.labels_train = np.array(labels_train)
        self.features_test = np.array(features_test)
        self.labels_test = np.array(labels_test)
        self.ModelClass = ModelClass

    def objective(self, trial: optuna.trial.Trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 15)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical(
            "algorithm", ["ball_tree", "kd_tree", "brute"]
        )
        leaf_size = trial.suggest_int("leaf_size", 5, 60)
        p = trial.suggest_int("p", 1, 2)  # Manhattan vs Euclidean distances.

        model = self.ModelClass(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )
        model.fit(self.features_train, self.labels_train)
        score = model.score(self.features_test, self.labels_test)
        return score


def search(target: str, slice_sizes: List[int] = None, sat_cuts: List[float] = None):
    assert target in ["group", "score"]

    ModelClass = ImageVectorizer if target == "group" else KNeighborsRegressor
    logger.info(f"using {ModelClass.__name__} for {target}")

    # Load the metadata
    with open(OUT_DIR / "metadata.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        metadata = list(reader)
    logger.info("got metadata")

    tgt_idx = 3  # Default to "score".
    if target == "group":
        tgt_idx = 2
    labels: List[str] = [float(meta[tgt_idx]) for meta in metadata]

    best_best_params = [-1, -1, dict()]
    best_best_score = -1
    best_best_features = []
    best_best_labels = []

    for n in slice_sizes or [25, 75, 150]:
        for s in sat_cuts or [0.35, 0.5, 0.65]:
            # Load the images and extract the features
            partial_load = partial(load_features, n_color_slices=n, sat_cut=s)
            with Pool() as p:
                sub_img_filenames = [
                    sub_img_filename for sub_img_filename, _, _, _ in metadata
                ]
                features = p.map(partial_load, sub_img_filenames)
                logger.info("got features")

            # Split the data into a training set and a test set
            features_train, features_test, labels_train, labels_test = train_test_split(
                # Use a very limited amount of train data to control overfitting.
                features,
                labels,
                test_size=0.85,
                random_state=RANDOM_STATE,
            )

            # Use optuna to find the best parameters
            trainer = Trainer(
                features_train, labels_train, features_test, labels_test, ModelClass
            )
            study = optuna.create_study(direction="maximize")
            study.optimize(trainer.objective, n_trials=30)

            # Plot optimization history
            # fig = optuna.visualization.plot_optimization_history(study)
            # fig.show()

            # Plot parameter importance
            # fig = optuna.visualization.plot_param_importances(study)
            # fig.show()

            # Train the model with the best parameters
            model = ModelClass(**study.best_params)
            model.fit(features_train, labels_train)

            # Test the model
            model_score = model.score(features_test, labels_test)
            logger.info(f"Test score: {model_score} @ {n} slices / {s} sat_cut")
            if model_score > best_best_score:
                logger.info(
                    f"BEST: {model_score} @ {n} slices / {s} sat_cut\n\t{study.best_params}"
                )
                best_best_score = model_score
                best_best_params = [n, s, study.best_params]
                best_best_features = features_train
                best_best_labels = labels_train

    logger.info(
        f"BEST: {best_best_score} @ {best_best_params[0]}, {best_best_params[1]} slices\n\t{best_best_params[2]}"
    )
    # Train the model with the best parameters
    model = ModelClass(**(best_best_params[2]))
    model.fit(best_best_features, best_best_labels)
    # Save the model.
    joblib.dump(model, str(ARTIFACTS_DIR / f"model_{target}_best.joblib"))
    # Save the params.
    model_params = {
        "n_color_slices": best_best_params[0],
        "sat_cut": best_best_params[1],
        **best_best_params[2],
    }
    with open(
        ARTIFACTS_DIR / f"model_{target}_best.params.json",
        "w",
    ) as f:
        json.dump(model_params, f)


if __name__ == "__main__":
    # search("group")
    with open(ARTIFACTS_DIR / "model_group_best.params.json", "r") as f:
        model_params = json.load(f)
    # Use the proven, general purpose vector features from training the classifier.
    search("score", [model_params["n_color_slices"]], [model_params["sat_cut"]])
