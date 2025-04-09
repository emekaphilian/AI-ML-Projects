# Synthetic Chest X-ray Image Generator Using GAN

This project leverages Generative Adversarial Networks (GANs) to generate synthetic chest X-ray images for the purpose of privacy-preserving medical AI model training. The aim is to help researchers and developers access realistic, anonymized medical image data.

## 📂 Dataset

The dataset used is the "Chest X-ray Images (Pneumonia)" dataset from Kaggle:
- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

It contains:
- **Training**, **Validation**, and **Testing** sets
- Two classes: `NORMAL` and `PNEUMONIA`

## 🧪 Project Structure

```bash
├── Chest_Xray_GAN_Project.ipynb   # Full Colab notebook with training, evaluation, and results
├── streamlit_app.py               # Streamlit app to generate and visualize images
├── models/                        # Trained Generator model
├── outputs/                       # Generated images and sample grids
└── README.md                      # This file
```

## 🔍 Key Features

- **Preprocessing:** Images resized to 28x28 grayscale and normalized to [-1, 1]
- **GAN Architecture:** Custom Generator and Discriminator built using TensorFlow/Keras
- **Loss Visualization:** Generator and Discriminator loss curves plotted
- **Sample Grids:** Images generated during training for visual inspection
- **Evaluation:** FID score for quality assessment
- **App Deployment:** Streamlit app to interact with the generator
- **Export Options:** Project exported as HTML, PDF, and GitHub-ready

## 🚀 Running Locally

```bash
# Install dependencies
pip install streamlit tensorflow matplotlib numpy pillow

# Run the app
streamlit run streamlit_app.py
```

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Matplotlib
- Streamlit
- NumPy

## 📈 Results

- FID Score: ~[insert value here]
- Training stabilized around epoch [insert epoch]
- Generator able to produce distinguishable synthetic images from noise

## 📄 Conclusion

This project demonstrates how GANs can be used to create realistic medical data for privacy-focused AI training. It serves as a foundational step towards building privacy-preserving synthetic datasets in healthcare.

## 💼 Author

**Emeka Ogbonna**  
[LinkedIn](https://www.linkedin.com/in/emeka-ogbonna-946828225/) | ogbonnaemeka665@gmail.com

---
*For educational purposes only. Ensure any synthetic data generation complies with relevant privacy laws and institutional policies.*


