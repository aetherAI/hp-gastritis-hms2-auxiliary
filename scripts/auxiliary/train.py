import argparse
import csv
import pickle
import sys

import numpy as np
from flaml import AutoML


sys.path.append(".")

from src.logistic_regression import LogisticRegression  # noqa: E402
from src.utils import parse_config  # noqa: E402


def main() -> None:
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="A config file for an auxiliary model",
        required=True,
    )
    args = parser.parse_args()

    # Read the configs
    config = parse_config(args.config)

    # Get the datalists
    annotated_train_list = []
    with open(config["ANNOTATED_TRAIN_LIST"]) as file_:
        reader = csv.reader(file_)
        for row in reader:
            annotated_train_list.append(row[0])

    negative_train_list = []
    with open(config["TRAIN_CSV_PATH"]) as file_:
        reader = csv.reader(file_)
        for row in reader:
            if row[1] == "0":
                negative_train_list.append(row[0])

    # Get the train matrix
    train_matrix = np.load(config["TRAIN_MATRIX_PATH"])
    embs = train_matrix["embs"]
    labels = train_matrix["labels"]
    print(f"{config['TRAIN_MATRIX_PATH']} was loaded.")

    if config["AUXILIARY_MODEL_TYPE"] == "flaml":
        if config["FLAML_REBALANCE"]:
            positive_rate = labels.astype(float).mean()
            sample_weight = np.where(labels == 1, 1.0 - positive_rate, positive_rate)
        else:
            sample_weight = None

        # AutoML
        automl = AutoML()
        automl.fit(
            embs,
            labels,
            metric=config["FLAML_METRIC"],
            task="classification",
            n_jobs=16,
            estimator_list=["lgbm"],
            sample_weight=sample_weight,
            time_budget=config["FLAML_TIME_BUDGET"],
            n_splits=5,
            eval_method="cv",
            split_type="stratified",
        )

        with open(config["FLAML_DUMP_PATH"], "wb") as file_:
            pickle.dump(automl, file_, pickle.HIGHEST_PROTOCOL)

        automl.model.model.booster_.save_model(config["AUXILIARY_MODEL_PATH"])

        print(f"The model is saved as {config['AUXILIARY_MODEL_PATH']}")

    elif config["AUXILIARY_MODEL_TYPE"] == "logistic_regression":
        model = LogisticRegression(
            features=2048,
            max_iter=1000,
            l2_strength=config["LOGISTIC_REGRESSION_L2_STRENGTH"],
        )
        model.fit(embs, labels)
        model.save(config["AUXILIARY_MODEL_PATH"])
        print(f"The model is saved as {config['AUXILIARY_MODEL_PATH']}")
    else:
        raise ValueError(
            f"Unknown AUXILIARY_MODEL_TYPE: {config['AUXILIARY_MODEL_TYPE']}"
        )


if __name__ == "__main__":
    main()
