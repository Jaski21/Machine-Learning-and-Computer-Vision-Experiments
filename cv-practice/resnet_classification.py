import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def load_image(path: str) -> Image.Image:
    image_file = Path(path)
    img = cv2.imread(str(image_file))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")

    # OpenCV loads BGR; pretrained ImageNet weights expect RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def main(image_paths: list[str]) -> None:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    labels = weights.meta["categories"]

    for image_path in image_paths:
        image = load_image(image_path)
        image_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(image_tensor)

        probabilities = F.softmax(logits, dim=1)
        top_prob, top_class = probabilities.topk(1)

        predicted_label = labels[top_class.item()]
        confidence = top_prob.item()

        print(f"Image: {Path(image_path).name}")
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Output shape: {tuple(logits.shape)}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify images with a pretrained ResNet-18 model."
    )
    parser.add_argument("images", nargs="+", help="One or more image paths.")
    args = parser.parse_args()
    main(args.images)
