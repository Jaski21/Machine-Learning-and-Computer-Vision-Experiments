import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def main(image_path: str) -> None:
    image_file = Path(image_path)

    img = cv2.imread(str(image_file))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_color = cv2.resize(img, (224, 224))
    resized_gray = cv2.resize(gray, (224, 224))

    print("Original shape:", img.shape)
    print("Resized color shape:", resized_color.shape)
    print("Gray shape:", gray.shape)
    print("Resized gray shape:", resized_gray.shape)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Grayscale Image")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Resized Color Image")
    plt.imshow(cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Resized Grayscale Image")
    plt.imshow(resized_gray, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic image preprocessing with OpenCV.")
    parser.add_argument("image", help="Path to an image file.")
    args = parser.parse_args()
    main(args.image)
