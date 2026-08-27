import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


def load_image(path: str) -> Image.Image:
    image_file = Path(path)
    img = cv2.imread(str(image_file))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def main(image_path: str) -> None:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    image = load_image(image_path)
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)["out"]

    mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
    unique_classes = np.unique(mask)

    print("Detected classes:")
    for class_id in unique_classes:
        print(f"  {class_id}: {categories[class_id]}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="tab20")
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation with a pretrained DeepLabV3-ResNet50 model."
    )
    parser.add_argument("image", help="Path to an image file.")
    args = parser.parse_args()
    main(args.image)
