# CIFAR-10 - Object Recognition in Images

This project performs object recognition on images using the CIFAR-10 dataset. It includes complete steps from data extraction, preprocessing, and visualization, to training deep learning models including a basic neural network and ResNet50.

---

## 📦 Dataset

**Source:** [Kaggle - CIFAR-10](https://www.kaggle.com/c/cifar-10)

- 60,000 32x32 color images in 10 classes
- 50,000 training images, 10,000 test images
- Classes:
  - `airplane`, `automobile`, `bird`, `cat`, `deer`
  - `dog`, `frog`, `horse`, `ship`, `truck`

---

## 🛠 Tech Stack

- **Languages:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, OpenCV, PIL
- **ML/DL:** TensorFlow, Keras
- **Tools:** Google Colab, Kaggle API, Py7zr

---

## 📂 Project Structure

```

cifar10-object-recognition/
├── cifar-10.zip
├── train.7z
├── test.7z
├── trainLabels.csv
├── sampleSubmission.csv
├── kaggle.json
├── train/
├── src/
│   ├── preprocess.py
│   ├── train\_simple\_nn.py
│   ├── train\_resnet.py
├── README.md
└── requirements.txt

````

---

## ⚙️ Setup Instructions

1. **Install Kaggle CLI and upload your `kaggle.json`**

```python
from google.colab import files
files.upload()  # upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
````

2. **Download CIFAR-10 dataset**

```bash
!kaggle competitions download -c cifar-10
```

3. **Extract dataset files**

```python
from zipfile import ZipFile
with ZipFile('/content/cifar-10.zip', 'r') as zip_ref:
    zip_ref.extractall()
```

4. **Extract `.7z` files**

```python
!pip install py7zr
import py7zr
with py7zr.SevenZipFile('/content/train.7z', mode='r') as z:
    z.extractall(path='train')
```

---

## 🧪 Model Training

### 1. Basic Neural Network

* Flatten → Dense(64, ReLU) → Dense(10, Softmax)
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy
* Accuracy after 10 epochs: \~36%

### 2. Transfer Learning (ResNet50)

* ResNet50 base (ImageNet weights)
* Custom classification head for 10 classes
* Input shape resized to (256, 256, 3)

---

## 📊 Results

* Accuracy (Basic NN): \~36%
* Accuracy (ResNet50): Higher, varies by epochs and data augmentation
* Data normalized to \[0, 1]

---

## 📌 Notes

* Images read using `cv2.imread()` or `PIL.Image`
* Labels mapped using custom dictionary
* Used train-test split (80-20) via scikit-learn

---

