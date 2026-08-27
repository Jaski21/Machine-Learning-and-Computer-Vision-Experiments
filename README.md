# Machine Learning & Computer Vision Experiments

A collection of hands-on Python experiments exploring NumPy, fundamental machine learning concepts, and computer vision.

These projects were written as learning exercises rather than coursework submissions. The repository documents my progression from learning the foundations of numerical computing with NumPy, to implementing machine learning algorithms from scratch, and finally applying pretrained deep-learning models to computer vision tasks.

## Contents

### NumPy Fundamentals

`numpyPractice/`

A collection of small experiments covering the core NumPy concepts used throughout machine learning and scientific computing.

* `Basics.py` — Creating NumPy arrays and comparing NumPy arrays with Python lists.
* `Arithmetic.py` — Scalar operations, vectorized functions, and element-wise array operations.
* `AggregateFuncs.py` — Aggregate and statistical functions such as sum, mean, standard deviation, variance, minimum, maximum, and `argmax`.
* `Broadcasting.py` — Demonstrates NumPy broadcasting between arrays with different shapes.
* `Filtering.py` — Boolean indexing, conditional filtering, and `np.where`.
* `MultidimensionalArray.py` — Working with multidimensional arrays, dimensions, shapes, and indexing.
* `Random.py` — Random number generation using NumPy's random-number utilities.
* `Shuffle.py` — Shuffling arrays and randomly selecting elements.
* `Slicing.py` — Array slicing, row and column selection, and dividing a matrix into regions.

### Machine Learning

`ml-experiments`

The machine-learning experiments build on the NumPy fundamentals and implement linear regression without relying on a high-level machine-learning library.

* `single_variable.py` — Linear regression with one feature, including a hand-written cost function and gradient descent implementation.
* `multiple_variables.py` — Linear regression with multiple features using vector-valued weights and gradient descent.
* `multiple_variables_feature_scaling.py` — Extends multiple-variable linear regression with feature standardization and visualization of the scaled features.

### Computer Vision

`cv-practice/`

Experiments exploring image preprocessing and pretrained deep-learning models for computer vision.

* `image_preprocessing.py` — Loads an image with OpenCV, converts it to grayscale, resizes it, and visualizes the results.
* `resnet_classification.py` — Runs a pretrained ResNet-18 model on user-supplied images and reports the top ImageNet prediction and confidence.
* `deeplabv3_segmentation.py` — Runs a pretrained DeepLabV3-ResNet50 model and visualizes the resulting pixel-level semantic segmentation mask.

## Technologies

* Python
* NumPy
* Matplotlib
* OpenCV
* PyTorch
* TorchVision

## Project Structure

```text
ML-CV-Experiments/
├── numpyPractice/
│   ├── Basics.py
│   ├── Arithmetic.py
│   ├── AggregateFuncs.py
│   ├── Broadcasting.py
│   ├── Filtering.py
│   ├── MultidimensionalArray.py
│   ├── Random.py
│   ├── Shuffle.py
│   └── Slicing.py
│
├── ml-experiments/
│   ├── single_variable.py
│   ├── multiple_variables.py
│   └── multiple_variables_feature_scaling.py
│
├── cv-practice/
│   ├── image_preprocessing.py
│   ├── resnet_classification.py
│   └── deeplabv3_segmentation.py
│
├── assets/
│   └── images/
│
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The pretrained TorchVision models may download their weights the first time they are run.

## Running the Experiments

### NumPy Fundamentals

The NumPy exercises can be run individually. For example:

```bash
python numpyPractice/Basics.py
python numpyPractice/Arithmetic.py
python numpyPractice/AggregateFuncs.py
python numpyPractice/Broadcasting.py
python numpyPractice/Filtering.py
python numpyPractice/MultidimensionalArray.py
python numpyPractice/Random.py
python numpyPractice/Shuffle.py
python numpyPractice/Slicing.py
```

These scripts are intentionally small and are designed to demonstrate individual NumPy concepts.

### Linear Regression

Run the single-variable implementation:

```bash
python ml-experiments/linear-regression/single_variable.py
```

Run the multiple-variable implementation:

```bash
python ml-experiments/linear-regression/multiple_variables.py
```

Run the feature-scaling experiment:

```bash
python ml-experiments/linear-regression/multiple_variables_feature_scaling.py
```

The single-variable and feature-scaling experiments produce Matplotlib visualizations.

### Computer Vision

Place your own test images in:

```text
assets/images/
```

Run the image-preprocessing experiment:

```bash
python cv-practice/image_preprocessing.py assets/images/image.jpg
```

Run ResNet-18 image classification:

```bash
python cv-practice/resnet_classification.py assets/images/image.jpg
```

The ResNet script also accepts multiple images:

```bash
python cv-practice/resnet_classification.py assets/images/Dog.jpg assets/images/TV.jpg assets/images/UI.jpg
```

Run DeepLabV3 semantic segmentation:

```bash
python cv-practice/deeplabv3_segmentation.py assets/images/interior.jpg
```

The NumPy exercises establish the array manipulation and vectorized-computation fundamentals used in later experiments.

The linear regression implementations then build on those foundations by explicitly implementing the cost function, gradients, and parameter updates used by gradient descent.

The computer vision experiments extend this progression into deep learning by working with image tensors, preprocessing pipelines, pretrained neural networks, and pixel-level predictions.

## Notes

* The NumPy exercises are intentionally small and focus on learning individual concepts rather than solving a single application.
* The linear regression implementations deliberately keep the optimization logic explicit instead of hiding it behind a high-level machine-learning library.
* The image files used during the original computer-vision experiments are not included in this repository. Add your own test images to `assets/images/`.
* The computer-vision scripts use the preprocessing associated with the pretrained TorchVision weights.
* OpenCV loads images in BGR order, so the computer-vision scripts convert images to RGB before displaying them or passing them to pretrained models.
* `torch.no_grad()` is used during inference to disable gradient tracking because the model parameters are not being updated.
* The classification and segmentation models are pretrained models; they are not trained from scratch in this repository.
