import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch


CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "dataset" / "mnist.pkl.gz"


def show_samples(data_path, sample_count=10):
    with gzip.open(data_path, "rb") as file:
        _, _, test_data = pickle.load(file, encoding="latin1")

    images, labels = test_data
    sampled_images = torch.tensor(images[:sample_count], dtype=torch.float32)
    sampled_labels = torch.tensor(labels[:sample_count], dtype=torch.long)
    torch.save(
        {"images": sampled_images, "labels": sampled_labels},
        CURRENT_DIR / "mnist_samples.pt",
    )

    figure, axes = plt.subplots(2, 5, figsize=(10, 5))

    for axis, image, label in zip(
        axes.flat,
        images[:sample_count],
        labels[:sample_count],
    ):
        axis.imshow(image.reshape(28, 28), cmap="gray")
        axis.set_title(f"Label: {label}")
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(CURRENT_DIR / "mnist_samples.png")
    plt.show()


if __name__ == "__main__":
    show_samples(DATA_PATH)
