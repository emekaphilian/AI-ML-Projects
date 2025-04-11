
# 🧠 Fashion MNIST Image Classification with TensorFlow & Keras

Welcome to my deep learning project on **Fashion MNIST image classification**! This project demonstrates how to build, train, evaluate, and interact with a Convolutional Neural Network (CNN) model using **TensorFlow/Keras** on a widely-used dataset of fashion items. The final model allows users to **input an image index** and receive real-time predictions on what the image represents.

---

## 🚀 Project Overview

- **Dataset**: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale 28x28 images in 10 categories
- **Goal**: Classify clothing items such as T-shirts, trousers, shoes, and bags using CNNs
- **Tools & Frameworks**: Python, TensorFlow, Keras, NumPy, Matplotlib

---

## 🧰 Tech Stack

- Python 3.x
- TensorFlow / Keras
- Matplotlib
- NumPy

---

## 🖼️ Dataset Details

The Fashion MNIST dataset includes:
- 60,000 training images
- 10,000 test images
- 10 total fashion categories:

| Label | Class        |
|-------|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |

---

## 🧠 Model Architecture

- Input Layer: 28x28 grayscale images
- 2 Convolutional Layers + MaxPooling
- Flatten + Dense Layers
- Output Layer with 10 softmax neurons

The model is trained using **Sparse Categorical Crossentropy** and **Adam optimizer** with high accuracy on validation data.

---

## ✨ Interactive Prediction

At the end of the script, you’ll find an **interactive section** where you can:
- Input an index (0–9999)
- View the corresponding image
- Get a prediction of the fashion category

```bash
Enter an image index (0-9999) to predict: 42
```

It returns the predicted label and confidence score — e.g., “Sneaker (Confidence: 97.35%)”.

---

## 📁 Files in This Repo

- `image_classification_mnist_fashion_interactive.py` — Full Python script with interactive section
- `README.md` — This file explaining the project structure

---

## ✅ How to Run

1. Clone this repo:
```bash
git clone https://github.com/your-username/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

2. Install dependencies:
```bash
pip install tensorflow matplotlib numpy
```

3. Run the script:
```bash
python image_classification_mnist_fashion_interactive.py
```

---

## 📸 Sample Output

![Sample Fashion MNIST Image](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

---

## 🤝 Let's Connect!

Feel free to check out more of my work or connect on [LinkedIn](https://www.linkedin.com/in/emeka-ogbonna-946828225/).

---

## ⭐ Credits
- Project by Emeka Ogbonna
- Dataset by Zalando Research
- Built using TensorFlow and Keras
