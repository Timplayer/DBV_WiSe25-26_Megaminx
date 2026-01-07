from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "colors.csv"

RGB = Tuple[int, int, int]
ColorInput = Union[str, Iterable[int]]


def _clean_label(label: str) -> str:
    return label.strip().strip('"')


def _hex_to_rgb(value: str) -> RGB:
    hex_value = value.strip().strip('"').lstrip("#")
    if len(hex_value) != 6:
        raise ValueError(f"Expected 6-char hex color, got {value!r}")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def _to_rgb(color: ColorInput) -> RGB:
    if isinstance(color, str):
        return _hex_to_rgb(color)
    rgb = tuple(int(c) for c in color)
    if len(rgb) != 3:
        raise ValueError(f"Expected 3-channel RGB, got {color!r}")
    return rgb[0], rgb[1], rgb[2]


def load_color_samples(csv_path: Path = DEFAULT_DATA_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X: list[RGB] = []
    y: list[str] = []

    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        labels = [_clean_label(h) for h in header]

        for row in reader:
            for label, cell in zip(labels, row):
                if not cell:
                    continue
                X.append(_hex_to_rgb(cell))
                y.append(label)

    return np.array(X, dtype=np.float32), np.array(y), labels


def train_model(
    csv_path: Path = DEFAULT_DATA_PATH,
    n_neighbors: int = 5,
) -> tuple[Pipeline, list[str]]:
    X, y, labels = load_color_samples(csv_path)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )
    model.fit(X, y)
    return model, labels


_MODEL: Optional[Pipeline] = None


def get_model(csv_path: Path = DEFAULT_DATA_PATH) -> Pipeline:
    global _MODEL
    if _MODEL is None:
        _MODEL, _ = train_model(csv_path=csv_path)
    return _MODEL


def classify(color: ColorInput, model: Optional[Pipeline] = None) -> str:
    model = model or get_model()
    rgb = _to_rgb(color)
    return str(model.predict([rgb])[0])


def classify_proba(
    color: ColorInput,
    model: Optional[Pipeline] = None,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    model = model or get_model()
    rgb = _to_rgb(color)
    proba = model.predict_proba([rgb])[0]
    classes = model.classes_
    ranked = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
    return [(str(label), float(score)) for label, score in ranked[:top_k]]


if __name__ == "__main__":
    model, labels = train_model()
    print(f"Trained on {len(labels)} labels from {DEFAULT_DATA_PATH}")
    sample = "0273F3"
    print(f"{sample} -> {classify(sample, model)}")


