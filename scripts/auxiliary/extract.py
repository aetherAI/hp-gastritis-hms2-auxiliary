import argparse
import csv
import os
import sys

import numpy as np
import torch
from hms2.core.builder import Hms2ModelBuilder
from hms2.pipeline.official_openslide import BoundingBox, OfficialOpenSlideReader
from hms2.pipeline.utils import torch_load
from PIL import Image
from tqdm import tqdm


sys.path.append(".")

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
    hms_config = parse_config(config["HMS_CONFIG_RECORD_PATH"])

    # Get the datalists
    train_list = []
    with open(config["ANNOTATED_TRAIN_LIST"]) as file_:
        reader = csv.reader(file_)
        for row in reader:
            train_list.append(row[0])
    with open(config["TRAIN_CSV_PATH"]) as file_:
        reader = csv.reader(file_)
        for row in reader:
            if row[1] == "0":
                train_list.append(row[0])

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

    # Read the embeddings and labels
    print("Extracting embeddings")
    embs = []
    labels = []
    slides = []
    model_emb.eval()
    for slide in tqdm(train_list):
        # Compute the embedding feature map
        reader = OfficialOpenSlideReader(
            os.path.join(
                config["SLIDE_DIR"],
                f"{slide}{config['SLIDE_FILE_EXTENSION']}",
            ),
        )
        box = BoundingBox(left=0, top=0, width=reader.width, height=reader.height)
        image = reader.get_region(box, scale=config["RESIZE_RATIO"], padding=False)
        with torch.no_grad():
            embedding_map = (
                model_emb(
                    torch.tensor(image)[np.newaxis, ...],
                )[0]
                .permute((1, 2, 0))
                .numpy()
            )

        # Get the mask
        mask = np.array(
            Image.open(os.path.join(config["MASK_FOLDER"], f"{slide}.tiff")),
        )
        mask = mask[: embedding_map.shape[0], : embedding_map.shape[1]]
        dont_care = mask == config["MASK_DONTCARE_VALUE"]
        embs.append(embedding_map[~dont_care])
        labels.append(
            np.where(
                mask[~dont_care] == config["MASK_POSITIVE_VALUE"],
                1,
                0,
            ).astype(int),
        )
        slides.append([slide] * (~dont_care).astype(int).sum())

    embs = np.concatenate(embs, axis=0)
    labels = np.concatenate(labels, axis=0)
    slides = np.concatenate(slides, axis=0)
    print(f"Positive samples: {(labels == 1).astype(int).sum()}")
    print(f"Negative samples: {(labels == 0).astype(int).sum()}")

    np.savez(config["TRAIN_MATRIX_PATH"], embs=embs, labels=labels, slides=slides)
    print(
        f"The training matrix was saved as {config['TRAIN_MATRIX_PATH']}",
    )


if __name__ == "__main__":
    main()
