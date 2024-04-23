import csv
import io
import logging
import os
import pathlib
import re
from multiprocessing import Pool, cpu_count
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_DIR = pathlib.Path("../data")
OUT_DIR = DATA_DIR / "subdata/"

CSV_SEP = ","
CSV_HEADERS = ["image_filename", "parent", "group"]
CSV_OUT_PATH = OUT_DIR / "metadata.csv"

# Convolution parameters.
SUBIMAGE_SIZE = 1024
STEP_SIZE = 512
DOWNSAMPLE_SIZE = 128  # For memory considerations.


def extract_parent(s: str) -> str:
    return re.split(r"\.+", s)[0]


def extract_group(s: str) -> str:
    loc = re.split(r"[-_ .]+", s)[0]
    city = re.sub(r"[0-9]", "", loc)
    return city


def process_image(filename):
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=CSV_SEP, strict=True)

    if filename.endswith(".tif"):
        current_image = os.path.basename(filename)
        img = Image.open(filename)
        width, height = img.size

        image_group = extract_group(current_image)
        image_parent = extract_parent(current_image)
        logging.info(f"splitting up {image_parent} (group: {image_group})...")

        remainder_width = width % STEP_SIZE
        remainder_height = height % STEP_SIZE
        left = remainder_width // 2
        upper = remainder_height // 2
        right = width - (remainder_width - left)
        lower = height - (remainder_height - upper)
        img = img.crop((left, upper, right, lower))
        width, height = img.size

        count = 0
        for i in range(0, width, STEP_SIZE):
            for j in range(0, height, STEP_SIZE):
                if i + SUBIMAGE_SIZE <= width and j + SUBIMAGE_SIZE <= height:
                    box = (i, j, i + SUBIMAGE_SIZE, j + SUBIMAGE_SIZE)
                    sub_img = img.crop(box).resize((DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE))

                    sub_img_filename = f"{image_parent}_{i}_{j}.tif"
                    sub_img.save(OUT_DIR / sub_img_filename)
                    writer.writerow([sub_img_filename, image_parent, image_group])
                count += 1
        logging.info(f"\t{count} sub-images written")

    return buf.getvalue()


def generate():
    pool = Pool(cpu_count())  # Create a multiprocessing Pool

    image_files = []
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        if "images" in dirpath:
            for filename in filenames:
                image_files.append(os.path.join(dirpath, filename))

    results = pool.map(process_image, image_files)

    with open(CSV_OUT_PATH, "w") as f:
        f.write(CSV_SEP.join(CSV_HEADERS) + "\n")
        for result in results:
            f.write(result)
    logging.info(f"wrote metadata to {os.path.basename(CSV_OUT_PATH)}")


if __name__ == "__main__":
    generate()
