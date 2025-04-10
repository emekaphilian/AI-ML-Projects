# Synthetic Chest X-ray Image Generation Using GANs

This project uses Generative Adversarial Networks (GANs) to generate synthetic chest X-ray images. It is designed for privacy-preserving medical AI applications, especially in contexts where patient data protection is critical.

## 🧠 Overview

The aim of this project is to explore the potential of GANs in creating high-quality synthetic medical images that can be used for training machine learning models without compromising patient privacy. It focuses on chest X-ray datasets and leverages deep learning techniques to build and train a GAN.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV
- Matplotlib
- Google Colab (for training)

## ⚙️ How It Works

1. **Data Preparation**: Chest X-ray images are loaded from a structured directory (`train`, `test`, `val`) using OpenCV and NumPy.
2. **GAN Architecture**: A custom GAN is implemented, consisting of:
   - Generator
   - Discriminator
3. **Training Loop**: The GAN is trained to generate realistic X-ray images over multiple epochs.
4. **Visualization**: Generated images are saved and displayed after training steps to monitor progress.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- TensorFlow >= 2.x
- OpenCV
- NumPy, Matplotlib, Pandas

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/synthetic-xray-gan.git
cd synthetic-xray-gan
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

To train the GAN:

```bash
python synthetic_xray_gan.py
```

Ensure that your chest X-ray dataset is structured and the path is correctly set in the script.

## 📁 Directory Structure

```
synthetic-xray-gan/
│
├── synthetic_xray_gan.py       # Main GAN training script
├── README.md                   # Project documentation
└── results/                    # (Optional) Generated images output
```

## 📊 Results

Sample output images will be generated and saved to disk after training. You can monitor model progress visually.

## 🙌 Credits

Inspired by research in privacy-preserving AI and the use of GANs for synthetic medical data generation.

## 📄 License

This project is open source and available under the MIT License.
