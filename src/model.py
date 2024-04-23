from functools import cache

import numpy as np
from PIL import Image
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from src.rgb_conv import rgb_to_hls

HSL_DIMS = 128


class ImageClassifier:
    def __init__(
        self,
        n_neighbors: int = 3,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
    ):
        self.knn: KNeighborsClassifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=-1,
        )
        self.le: LabelEncoder = LabelEncoder()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        labels = self.le.fit_transform(labels)
        self.knn.fit(features, labels)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        labels = self.le.transform(labels)
        return self.knn.score(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.le.inverse_transform(self.knn.predict(features))

    def save(self, model_path: str) -> None:
        dump((self.knn, self.le), model_path)

    @classmethod
    def load(cls, model_path: str) -> "ImageClassifier":
        knn, le = load(model_path)
        model = cls()
        model.knn = knn
        model.le = le
        return model

    @staticmethod
    def extract_features(
        image_path: str,
        n_color_slices: int = 150,
        sat_cut: float = 0.35,
    ) -> np.ndarray:
        """
        You can think of the HSL color space as a cylinder.
          * Hue is the angle around the central vertical axis
          * Saturation is the distance from this axis
          * Luminance is the height on the vertical axis

        This method limits considered hue values to those where the luminance is
        above 10% and the saturation is within the middle 90%.

        It then creates a two-dimensional histogram, slicing the hue-saturation
        cylinder radially into 50 sections (for 100 color slices) and horizontally
        into 2 sections. This is like a pie cut into wedges, then  cutting each
        wedge in half somewhere between the crust and pointy slice tip.

        Each cell in the histogram represents the proportion of pixels in the image
        that fall within the corresponding hue and luminance range.
        """
        hue_restricted, sat_restricted = get_hue_sat(image_path)

        # Create a 2D histogram of hue and sat values - dealing with normalized data, so add the range param.
        sat_bins = [0, sat_cut, 1]
        hist2d, _, _ = np.histogram2d(
            hue_restricted,
            sat_restricted,
            bins=(n_color_slices // 2, sat_bins),
            range=((0, 1), (0, 1)),
        )

        # Normalize histogram to create a proportion - the count vs total pixels.
        hist2d = hist2d.astype(float) / float(HSL_DIMS**2)

        # Flatten the 2D histogram to a 1D array.
        hist = hist2d.flatten()
        # And fill any NaN with zero.
        return np.nan_to_num(hist)


# --- HELPERS ---


@cache
def get_hue_sat(image_path: str):
    img_pil = Image.open(image_path)

    # Resizes image for lower features, then reshapes to one row per pixel, then normalizes from 0 to 1.
    rgb_data = np.array(img_pil.resize((HSL_DIMS, HSL_DIMS))).reshape(-1, 3) / 255.0
    # Convert RGB to HSL so we can slice easily.
    hsl_data = rgb_to_hls(rgb_data)

    # Extract HSL dimensions.
    hue = hsl_data[:, 0]
    sat = hsl_data[:, 1]
    luminance = hsl_data[:, 2]

    # Ignore hue where luminance is below 10% or saturation is not in middle 90%.
    hue_restricted = hue[(luminance > 0.1) | (sat > 0.05) | (sat > 0.95)]
    sat_restricted = sat[(luminance > 0.1) | (sat > 0.05) | (sat > 0.95)]
    return hue_restricted, sat_restricted
