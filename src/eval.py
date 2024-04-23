import logging
import os
import pathlib
import re

import joblib
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor

from src.vectorizer import ImageVectorizer

logging.basicConfig(level=logging.INFO)

SAMPLE_SCORES = {
    "condesa-cdmx": 88,
    "fresno": 50,
    "ft-worth": 40,
    "north-platte": 30,
    "parnell-auckland": 82,
    "sacile": 90,
}

SUBIMAGE_SIZE = 128
STEP_SIZE = 128
DOWNSAMPLE_SIZE = 128

ARTIFACTS_DIR = pathlib.Path("../artifacts")
SAMPLES_DIR = pathlib.Path("../data/samples/")
TMP_DIR = pathlib.Path("../tmp/")


def extract_parent(s: str) -> str:
    return re.split(r"\.+", s)[0]


def extract_group(s: str) -> str:
    loc = re.split(r"[_ .]+", s)[0]
    city = re.sub(r"[0-9]", "", loc)
    return city


def process_image(
    filename: str, model: KNeighborsRegressor
) -> tuple[str, float] | tuple[str, int]:
    if filename.endswith(".png"):
        current_image = os.path.basename(filename)
        img = Image.open(filename)
        width, height = img.size

        image_group = extract_group(current_image)
        city_score = SAMPLE_SCORES[image_group]
        logging.info(f"predicting for {image_group}...")
        logging.info(f"TARGET: {city_score:.1f}")

        remainder_width = width % STEP_SIZE
        remainder_height = height % STEP_SIZE
        left = remainder_width // 2
        upper = remainder_height // 2
        right = width - (remainder_width - left)
        lower = height - (remainder_height - upper)
        img = img.crop((left, upper, right, lower))
        width, height = img.size

        scores = []
        filenames = []
        for i in range(0, width, STEP_SIZE):
            for j in range(0, height, STEP_SIZE):
                if i + SUBIMAGE_SIZE <= width and j + SUBIMAGE_SIZE <= height:
                    box = (i, j, i + SUBIMAGE_SIZE, j + SUBIMAGE_SIZE)
                    sub_img = img.crop(box).resize((DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE))
                    sub_img = sub_img.convert("RGB")

                    sub_img_filename = f"{image_group}_{i}_{j}.tif"
                    sub_img.save(TMP_DIR / sub_img_filename)
                    filenames.append(TMP_DIR / sub_img_filename)

                    scores.append(
                        model.predict(
                            [
                                ImageVectorizer.extract_features(
                                    str(TMP_DIR / sub_img_filename)
                                )
                            ]
                        )[0]
                    )

        logging.info(f"\t{len(filenames)} sub-images predicted")
        for idx, fn in enumerate(filenames):
            logging.info(f"\t{os.path.basename(fn)} - {scores[idx]:.1f}")
        agg_score = sum(scores) / len(scores)
        logging.info(f"\tAGGREGATED: {agg_score:.1f}")
        logging.info(f"\tERROR: {city_score - agg_score:.1f}")

        return image_group, agg_score
    return "", -100


if __name__ == "__main__":
    model = joblib.load(ARTIFACTS_DIR / "model_score_best.joblib")
    logging.info(f"using {model.__class__.__name__}")
    table = []
    for dirpath, _, filenames in os.walk(SAMPLES_DIR):
        for filename in filenames:
            table.append(process_image(os.path.join(dirpath, filename), model))
    print("\n***")
    for city, score in sorted(table, key=lambda x: x[0]):
        print(f"{city:12} : {int(score)}")
