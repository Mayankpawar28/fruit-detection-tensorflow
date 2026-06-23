# Fruit Classifier CNN

A Convolutional Neural Network (CNN) that identifies **100+ fruit varieties** from images, trained on the Fruits-360 dataset using TensorFlow and Keras. Upload any fruit image and get an instant prediction.

---

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Kaggle API (`kaggle`)
- Google Colab

Install dependencies:

```bash
pip install tensorflow numpy matplotlib kaggle
```

---

## Setup

### 1. Kaggle API Key
Place your `kaggle.json` in the Colab environment, then run:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Dataset

```bash
kaggle datasets download -d moltean/fruits
unzip -q fruits.zip -d fruits_data
```

Dataset structure after extraction:

```
fruits_data/
└── fruits-360_100x100/
    └── fruits-360/
        ├── Training/
        └── Test/
```

---

## Pipeline

### Step 1 — Preprocess Images
- Rescale pixel values to `[0, 1]`
- Resize all images to **100×100 pixels**
- Batch size: **32**
- Multi-class categorical labels (one folder per fruit)

### Step 2 — Build CNN Model

| Layer | Details |
|---|---|
| Conv2D + MaxPooling | 32 filters, 3×3, ReLU |
| Conv2D + MaxPooling | 64 filters, 3×3, ReLU |
| Conv2D + MaxPooling | 128 filters, 3×3, ReLU |
| Flatten | — |
| Dense | 256 units, ReLU |
| Dropout | 0.3 |
| Dense (output) | N classes, Softmax |

### Step 3 — Compile & Train
- **Loss:** Categorical cross-entropy
- **Optimizer:** Adam
- **Epochs:** 5

### Step 4 — Visualize Training
Plots training vs. validation accuracy and loss over epochs.

### Step 5 — Predict on a Custom Image
Upload any fruit image in Colab and the model outputs the predicted fruit class.

---

## Example Output

```
Predicted class: Apple Red 1
```

---

## Dataset

**Source:** [Fruits-360 on Kaggle](https://www.kaggle.com/datasets/moltean/fruits) by Mihai Oltean

**Size:** 90,000+ images across 100+ fruit and vegetable categories, 100×100 px
