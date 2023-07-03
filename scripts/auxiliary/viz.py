import argparse
import csv
import os
import sys

import lightgbm as lgb
import numpy as np
import torch
from hms2.core.builder import Hms2ModelBuilder
from hms2.pipeline.official_openslide import BoundingBox, OfficialOpenSlideReader
from hms2.pipeline.utils import torch_load


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

    # Read the config
    config = parse_config(args.config)
    hms_config = parse_config(config["HMS_CONFIG_RECORD_PATH"])

    # Get the datalist
    viz_list = []
    with open(config["VIZ_CSV_PATH"]) as file_:
        reader = csv.reader(file_)
        for row in reader:
            viz_list.append(row[0])

    # Get the HMS model to produce embeddings
    model_emb = Hms2ModelBuilder().build(
        n_classes=hms_config["NUM_CLASSES"],
        backbone=hms_config["MODEL"],
        pooling="no",
        custom_dense="no",
        use_hms2=hms_config["USE_HMS2"],
        use_cpu_for_dense=True,
    )
    state_dict = torch_load(hms_config["MODEL_PATH"], map_location="cpu")
    model_emb.load_state_dict(state_dict, strict=False)
    print(model_emb)

    # Get the auxiliary model to produce HP heatmap
    if config["AUXILIARY_MODEL_TYPE"] == "flaml":
        auxiliary_model = lgb.Booster(model_file=config["AUXILIARY_MODEL_PATH"])
    elif config["AUXILIARY_MODEL_TYPE"] == "logistic_regression":
        auxiliary_model = LogisticRegression(
            features=2048,
            max_iter=1000,
            l2_strength=config["LOGISTIC_REGRESSION_L2_STRENGTH"],
        )
        auxiliary_model.load(config["AUXILIARY_MODEL_PATH"])
    else:
        raise ValueError(
            f"Unknown AUXILIARY_MODEL_TYPE: {config['AUXILIARY_MODEL_TYPE']}"
        )

    # Initiation
    os.makedirs(config["VIZ_RESULT_FOLDER"], exist_ok=True)

    # Processing slides
    for slide in viz_list:
        print(f"Processing {slide}...")

        # Read the slide
        reader = OfficialOpenSlideReader(
            os.path.join(
                config["SLIDE_DIR"],
                f"{slide}{config['SLIDE_FILE_EXTENSION']}",
            ),
        )
        box = BoundingBox(left=0, top=0, width=reader.width, height=reader.height)
        image = reader.get_region(box, scale=config["RESIZE_RATIO"], padding=False)

        # Use the HMS model to generate a embedding map
        with torch.no_grad():
            embedding_map = (
                model_emb(
                    torch.tensor(image)[np.newaxis, ...],
                )[0]
                .permute((1, 2, 0))
                .numpy()
            )  # [H, W, C]

        # Use the auxiliary model to generate a HP heatmap
        height, width, channels = embedding_map.shape
        embedding_map_flatten = embedding_map.reshape((height * width, channels))
        if config["AUXILIARY_MODEL_TYPE"] == "flaml":
            heatmap_flatten = auxiliary_model.predict(embedding_map_flatten)
        elif config["AUXILIARY_MODEL_TYPE"] == "logistic_regression":
            heatmap_flatten = auxiliary_model.predict_proba(embedding_map_flatten)

        heatmap = heatmap_flatten.reshape((height, width))
        print(f"Max value in the heatmap: {heatmap.max()}")

        # Save the HP heatmap
        path = os.path.join(config["VIZ_RESULT_FOLDER"], f"{slide}.npy")
        np.save(path, heatmap)


if __name__ == "__main__":
    main()
