import csv
import pathlib
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from model import ImageClassifier


from matplotlib.colors import ListedColormap

ARTIFACTS_DIR = pathlib.Path("../artifacts")
DATA_DIR = pathlib.Path("../data")
OUT_DIR = DATA_DIR / "subdata/"

# Make a custom color map.
# Arranged specifically to group certain "types" of city by color.
COLORS = [
    "#e6194b",  # red - Austin
    "#4363d8",  # blue - Bellingham
    "#46f0f0",  # cyan - Bloomington
    "#f58231",  # orange - Chicago
    "#3cb44b",  # green - Innsbruck
    "#911eb4",  # purple - Kitsap
    "#f032e6",  # magenta - SFO
    "#ffe119",  # yellow - Tyrol
    "#bcf60c",  # lime - Vienna
]
CUSTOM_COLORMAP = ListedColormap(COLORS, name="custom_cmap")


def sample_data(
    features: List[List[float]], labels: List[str], sample_size: float = 0.1
) -> Tuple[List[List[float]], List[str]]:
    """
    Randomly sample a fraction of the feature/label combinations.
    """
    num_samples = int(len(features) * sample_size)
    indices = random.sample(range(len(features)), num_samples)
    new_features = [features[i] for i in indices]
    new_labels = [labels[i] for i in indices]
    return new_features, new_labels


def fig_name(n_components: int, multi_angle: bool, sparsify: bool) -> str:
    name_components = [f"{n_components}d-pca"]
    if multi_angle and n_components > 2:
        name_components.append("multi-angle")
    if sparsify:
        name_components.append("sparsified")
    return "-".join(name_components) + ".png"


def plot_pca(
    features: List[List[float]],
    labels: List[str],
    n_components: int = 2,
    multi_angle: bool = True,
    sparsify: bool = False,
) -> None:
    if sparsify:
        features, labels = sample_data(features, labels)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    class_names = list(le.classes_)

    if multi_angle and n_components == 3:
        fig = plt.figure(figsize=(16, 10))
        angles = [(8 * x, 30 * x) for x in range(1, 5)]
        for i, (elev, azim) in enumerate(angles, start=1):
            ax = fig.add_subplot(2, 2, i, projection="3d")
            scatter = ax.scatter(
                principal_components[:, 0],
                principal_components[:, 1],
                principal_components[:, 2],
                c=encoded_labels,
                cmap=CUSTOM_COLORMAP,
                edgecolor="k",
            )
            scatter.set_clim(-0.5, len(le.classes_) - 0.5)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"View {i} (elev={elev}, azim={azim})")
            cbar = plt.colorbar(
                scatter,
                ticks=range(len(le.classes_)),
                label="Cities",
                ax=ax,
                pad=0.1,
            )
            cbar.ax.set_yticklabels(class_names)
    else:
        fig = plt.figure(figsize=(10, 5))
        if n_components == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
        scatter = ax.scatter(
            *[principal_components[:, n] for n in range(n_components)],
            c=encoded_labels,
            cmap=CUSTOM_COLORMAP,
            edgecolor="k",
        )
        scatter.set_clim(-0.5, len(le.classes_) - 0.5)
        cbar = plt.colorbar(
            scatter,
            ticks=range(len(le.classes_)),
            label="Cities",
        )
        cbar.ax.set_yticklabels(class_names)

    plt.savefig(ARTIFACTS_DIR / fig_name(n_components, multi_angle, sparsify))
    plt.show()


def main():
    # Load metadata for images.
    with open(OUT_DIR / "metadata.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        metadata = list(reader)

    features: List[List[float]] = []
    labels: List[str] = []

    # Load the images and extract the features...
    for sub_img_filename, _, group in metadata:
        features.append(
            # Use default model params.
            ImageClassifier.extract_features(str(OUT_DIR / sub_img_filename)).tolist()
        )
        labels.append(group)
    print("got features")

    # Plot our PCA in 2D.
    plot_pca(features, labels, n_components=2)

    # Plot our PCA in 3D.
    plot_pca(features, labels, n_components=3)

    # Plot our PCA in 2D sparsified.
    plot_pca(features, labels, n_components=2, sparsify=True)

    # Plot our PCA in 3D sparsified.
    plot_pca(features, labels, n_components=3, sparsify=True)


if __name__ == "__main__":
    main()
