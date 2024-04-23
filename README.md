# Geographic Precog

Let's see if aerial imagery can predict ____.

#### A Word on Memory

I have 48GB of RAM, and it wasn't enough.
To make this work I needed many GB of nVME swap... good luck!

#### Contents

1. [Python Setup](#python-setup)
2. [Getting Raw Data](#getting-raw-data)
3. [Generating Our Data](#generating-our-data)

---

## Python Setup

This project uses Python 3.12, but it would probably work with other versions of Python 3.

```sh
python3.12 -m venv ./venv/ && \
source ./venv/bin/activate && \
pip install -r requirements.txt
```

## Getting Raw Data

This analysis uses the [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/files/), available at the link on 22 Apr 2024.

1. Download the multipart 7z files by running (or referencing) [`scripts/download-data.sh`](./scripts/download-data.sh)
    * Alternative: `curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash`
    * But you shouldn't pipe scripts from the internet to your shell.
    * Make sure you're in the `./data/` directory.
2. Run or reference [`scripts/extract-data.sh`](./scripts/extract-data.sh) to extract the archive
3. Carry on

## Generating Our Data

We're going to make little sub-tiles from our images, and associate them with other information (including parent image and image group).

```sh
python scripts/generate-subimages.py
```

...
